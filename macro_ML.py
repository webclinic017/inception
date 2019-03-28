# imports
import sys, os
from utils.basic_utils import *
from utils.pricing import *
from utils.fundamental import *

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from sklearn.ensemble import forest

from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.validation import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.metrics import precision_score, roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

# environment variables
freq = '1d'
cuts = { '1d': [-1, -0.1, -.02, .02, .1, 1.] }
cut_range = cuts[freq]
fwd_ret_labels = ["bear", "short", "neutral", "long", "bull"]

benchSL, sectorSL, riskSL, rateSL, bondSL, commSL, currSL = \
    config['benchmarks'], config['sectors'], config['risk'], config['rates'], \
    config['bonds'], config['commodities'], config['currencies']

symbols_list = benchSL + sectorSL + riskSL + rateSL + bondSL + commSL + currSL

keep_bench = excl(benchSL, ['^STOXX50E', '^AXJO'])
keep_fx = excl(currSL, ['HKD=X', 'MXN=X', 'AUDUSD=X', 'NZDUSD=X', 'TWD=X', 'CLP=X', 'KRW=X'])
keep_sect = excl(sectorSL, ['SPY', 'QQQ', 'DIA', 'IWM', 'XLC', 'XLRE'])
keep_bonds = ['LQD', 'HYG']

include = riskSL + keep_bench + keep_sect + rateSL + keep_fx + keep_bonds
invert = ['EURUSD=X', 'GBPUSD=X']
incl_price = riskSL

bench = '^GSPC'
y_col = 'fwdReturn'
pred_fwd_windows = [20, 60, 120]
rate_windows = [20, 60]
sec_windows, stds = [5, 20, 60], 1

# utility functions
def create_ds(px_close, context):

    train_model = context['train_model']
    portion = context['portion']
    verbose = context['verbose']
    # average the return of the next periods
    # select only rows where Y variable is not null
    ds_idx = px_close.dropna(subset=[bench]).index

    df_large = pd.DataFrame()
    rate_ft_df = rate_feats(px_close[rateSL], rate_windows) # rate transforms
    df_large[rate_ft_df.columns] = rate_ft_df

    # price momentum transforms
    super_list = []
    for ticker in include:
        inv = ticker in invert
        incl_px = ticker in incl_price
        df = px_close[ticker]
        ft_df = px_mom_feats(df, ticker, stds, inv, incl_px, sec_windows)
        super_list.append(ft_df.drop_duplicates())
    df_large = pd.concat(super_list, axis=1).sort_index()
    df_large = df_large.loc[ds_idx, :] # drop NAs before discretizing

    if train_model:
        Y = px_fwd_rets(px_close.loc[ds_idx, bench], bench, pred_fwd_windows).mean(axis=1)
        df_large[y_col] = Y
        # reduce dataset?
        if portion < 100e-2: _, df_large = train_test_split(
            df_large, test_size=portion, random_state=42)
        if verbose:
            print('create_ds >> df_large.shape: ', df_large.shape)
            print('Y.shape: ', Y.shape)

    return ds_idx, df_large

def pre_process_ds(df, context):

    verbose = context['verbose']
    train_model = context['train_model']
    fill_on, imputer_on, scaler_on = \
        context['fill'], context['impute'], context['scale']
    test_sz = context['test_size']
    pred_batch = context['predict_batch']
    (path, train_cols) = context['trained_cols']

    imputer = SimpleImputer(
        missing_values=np.nan, strategy='median', copy=False)
    scaler = StandardScaler()
    X_cols = excl(df.columns, [y_col])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if fill_on: df.fillna(method=fill_on, inplace=True)
    if scaler_on: df.loc[:, X_cols] = scaler.fit_transform(df[X_cols])

    pred_X = df.iloc[-pred_batch:,:-1].copy() # how far back to predict
    X_train = X_test = y_train = y_test = None

    if train_model:
        if fill_on: df[X_cols].fillna(method=fill_on, inplace=True)
        # discretize forward returns into classes
        df.dropna(subset=[y_col], inplace=True)
        df[y_col] = discret_rets(df[y_col], cut_range, fwd_ret_labels)
        df[y_col] = df[y_col].astype(str)
        # this seems unnecesary when fill is on
        if imputer_on: df.loc[:, X_cols] = imputer.fit_transform(df[X_cols])
        else: df[X_cols].dropna(inplace=True)
        X, y = df.drop(columns=y_col), df[y_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_sz, random_state=42)
        np.save(path + train_cols, X_train.columns) # save feature order
        if verbose:
            y_col_dist = sample_wgts(df[y_col], fwd_ret_labels)
            print('pre_process_ds >> df_raw Y-var class distribution')
            print((y_col_dist[fwd_ret_labels]).round(3))

    return pred_X, X_train, X_test, y_train, y_test

def train_ds(context):

    print('Benchmark: {}, Y: {}, Include: {}, invert: {}, include price: {}'.format(
    bench, y_col, include, invert, incl_price))

    context['train_model'] = True
    ml_path = context['ml_path']
    grid_search = context['grid_search']
    verbose = context['verbose']

    px_close = get_mults_pricing(include, freq, verbose=verbose);
    px_close.drop_duplicates(inplace=True)
    if verbose: print('train_ds >> px_close.shape', px_close.shape)

    # create and pre-process datasets
    px_idx, df_raw = create_ds(px_close, context)
    pred_X, X_train, X_test, y_train, y_test = pre_process_ds(df_raw, context)
    if verbose:
        for x in zip(('df_raw', 'pred', 'X_train', 'y_train', 'X_test', 'y_test'),
        (df_raw, pred_X, X_train, y_train, X_test, y_test)):
            print(x[0] + '.shape', x[1].shape)

    # RandomForestClassifier
    best_params = {
        'max_features': 'sqrt', 'n_estimators': 100,
        'random_state': 4}
    if grid_search:
        print('GridSearchCV for RandomForestClassifier')
        param_grid = {
            'n_estimators': [100], 'max_features': ['sqrt'],
            'random_state': np.arange(0, 5, 1),}
        clf = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, n_jobs=-1,
                           cv=5, iid=True, verbose=verbose)
        clf.fit(X_train, y_train)
        if verbose: print_cv_results(
            clf, X_train, X_test, y_train, y_test,
            feat_imp=True, top=20)
        best_params = clf.best_params_
    clf1 = RandomForestClassifier(**best_params)
    clf1.fit(X_train, y_train)
    print('RandomForestClassifier scores: Train {}, Test {}'.format(
    clf1.score(X_train, y_train), clf1.score(X_test, y_test)))

    # MLPClassifier
    best_params = {
        'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': 95,
        'learning_rate': 'adaptive', 'max_iter': 200,
        'random_state': 4, 'solver': 'lbfgs'}
    if grid_search:
        print('GridSearchCV for MLPClassifier')
        param_grid = {
            'solver': ['lbfgs'], # ['lbfgs', 'sgd', 'adam']
            'max_iter': [200], # [200, 400, 600]
            'activation': ['relu'], # ['logistic', 'tanh', 'relu']
            'alpha': 10.0 ** -np.arange(2, 5, 1), # 10.0 ** -np.arange(2, 5, 1)
            'learning_rate' : ['adaptive'], # ['constant', 'adaptive']
            'hidden_layer_sizes': np.arange(5, X_train.shape[1] // 3, int(X_train.shape[1] * 0.1)), # np.arange(5, 50, 10)
            'random_state': np.arange(0, 5, 1)} # np.arange(0, 10, 2)
        clf = GridSearchCV(MLPClassifier(random_state=42), param_grid, n_jobs=-1, cv=5,
                          iid=True, verbose=verbose)
        clf.fit(X_train, y_train)
        if verbose: print_cv_results(
            clf, X_train, X_test, y_train, y_test,
            feat_imp=False, top=20)
        best_params = clf.best_params_

    clf2 = MLPClassifier(**best_params)
    clf2.fit(X_train, y_train)
    print('MLPClassifier scores Train {}, Test {}'.format(
    clf2.score(X_train, y_train), clf2.score(X_test, y_test)))

    # ExtraTreesClassifier
    clf3 = ExtraTreesClassifier(
        n_estimators=100, max_depth=None,
        min_samples_split=2, random_state=42)
    clf3.fit(X_train, y_train)
    print('ExtraTreesClassifier scores Train {}, Test {}'.format(
    clf3.score(X_train, y_train), clf3.score(X_test, y_test)))

    for vote in ['hard', 'soft']:
        eclf = VotingClassifier(
            estimators=[('rf', clf1), ('mlp', clf2), ('et', clf3)],
            voting=vote)
        clf = eclf.fit(X_train, y_train)
        print('VotingClassifier scores Train {}, Test {}'.format(
                clf.score(X_train, y_train), clf.score(X_test, y_test)))

        os.makedirs(ml_path, exist_ok=True)
        fname = ml_path + f'macro_ML_{vote}.pkl'
        joblib.dump(clf, fname)
        print('Saved ', fname)

def predict_ds(context):
    context['train_model'] = False
    ml_path = context['ml_path']
    verbose = context['verbose']
    (path, train_cols) = context['trained_cols']

    px_close = get_mults_pricing(include, freq, verbose=verbose);
    px_close.drop_duplicates(inplace=True)
    ds_idx, df_large = create_ds(px_close, context)
    pred_X, _, _, _, _ = pre_process_ds(df_large, context)
    print('pred_X.shape', pred_X.shape)

    # ensure prediction dataset is consistent with trained model
    trained_cols = np.load(path + train_cols) # save feature order
    missing_cols = [x for x in trained_cols if x not in pred_X.columns]
    pred_X = pd.concat([pred_X, pd.DataFrame(columns=missing_cols)], axis=1)
    pred_X[missing_cols] = 0
    pred_X = pred_X[list(trained_cols)]

    bench_df = px_close.loc[pred_X.index, bench].to_frame()
    for vote in ['hard', 'soft']:
        fname = ml_path + f'macro_ML_{vote}.pkl'
        clf = joblib.load(fname) # load latest models
        print('Loaded', fname)
        preds = clf.predict(pred_X)
        pred_class = np.array([fwd_ret_labels.index(x) for x in preds])
        bench_df[f'{vote}_pred_class'] = pred_class
        bench_df[f'{vote}_pred_label'] = preds
        if vote == 'soft':
            probs = clf.predict_proba(pred_X)
            pred_prob = np.argmax(probs, axis=1)
            bench_df[f'{vote}_confidence'] = [x[np.argmax(x)] for x in probs] # higest prob
            prob_df = pd.DataFrame(probs, index=bench_df.index, columns=clf.classes_)
            bench_df = pd.concat([bench_df, prob_df[fwd_ret_labels]], axis=1)
        bench_df.dropna(subset=[bench], inplace=True)

    # store in S3
    s3_df = bench_df.reset_index(drop=False)
    rename_col(s3_df, 'index', 'pred_date')
    csv_store(s3_df, 'recommend/', 'macro_risk_ML.csv')

    return bench_df

def print_cv_results(clf, X_train, X_test, y_train, y_test, full_grid=False, feat_imp=True, top=20):
    print(clf)
    cvres = clf.cv_results_
    print('BEST PARAMS:', clf.best_params_)
    print('SCORES:')
    print('clf.best_score_', clf.best_score_)
    print('train {}, test {}'.format(
        clf.score(X_train, y_train),
        clf.score(X_test, y_test)))
    if full_grid:
        print('GRID RESULTS:')
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(round(mean_score, 3), params)
    if feat_imp:
        feature_importances = clf.best_estimator_.feature_importances_
        print('SORTED FEATURES:')
        print(sorted(zip(feature_importances, list(X_train.columns)), reverse=True)[:top])

def visualize_predictions(pred_df):
    pre_class_cols = filter_cols(pred_df.columns, "pred_class")
    pred_df.loc[:,[bench] + pre_class_cols].plot(
        secondary_y=pre_class_cols, figsize=(15, 5));
    pred_df[fwd_ret_labels].plot.area(
            title='Prediction Probabilities',
            figsize=(15, 2), ylim=(0, 1), cmap='RdYlGn');
    f'Confidence Mean: {pred_df["soft_confidence"].mean().round(3)}, Median {pred_df["soft_confidence"].median().round(3)}'

def append_pricing(symbol, freq='1d', cols=None):
    """ appends most recent pricing to data on S3"""
    return appended_df

def pull_latest_px(tickers):
    """ get appended pricing from dataset """
    return px_close

#context/config for training and prediction
context = {
    'portion': 100e-2,
    'trained_cols': ('../ML/', 'macro_train_cols.npy'),
    'fill': 'bfill',
    'impute': True,
    'scale': True,
    'test_size': .20,
    'predict_batch': 252,
    'ml_path': './ML/',
    'grid_search': True,
    'verbose': 1}

if __name__ == '__main__':
    hook = sys.argv[1]
    if hook == 'train':
        print('TRAINING MACRO RISK EXPOSURE:')
        context['train_model'] = True
        train_ds(context)
    elif hook == 'predict':
        print('PREDICTING MACRO RISK EXPOSURE:')
        context['train_model'] = False
        pred_df = predict_ds(context)
        print(pred_df.tail(5).round(3).T)
    else: print('Invalid option, please try: train or predict')
