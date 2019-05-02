# imports
import sys, os
from utils.basic_utils import *
from utils.pricing import *
from utils.fundamental import *
from utils import ml_utils as mu

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from sklearn.ensemble import forest

from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.metrics import precision_score, roc_auc_score

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

# pricing, refresh once
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

# MODEL SPECIFIC FUNCTIONS
# utility functions
def create_ds(px_close, context):
    print('create_ds')
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

def pre_process_ds(raw_df, context):
    print('pre_process_ds')
    train_model = context['train_model']
    pred_batch = context['predict_batch']
    fill_on, imputer_on, scaler_on = context['fill'], context['impute'], context['scale']
    ml_path = context['ml_path']
    train_cols = context['trained_cols']
    test_sz, verbose = context['test_size'], context['verbose']

    scaler = StandardScaler()
    imputer = SimpleImputer(
        missing_values=np.nan,
        strategy='median', copy=False)
    X_cols = excl(raw_df.columns, [y_col])

    raw_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if scaler_on: raw_df[X_cols] = scaler.fit_transform(raw_df[X_cols])
    if fill_on: raw_df.loc[:, X_cols] = raw_df.fillna(method=fill_on)

    pred_X = X_train = X_test = y_train = y_test = None
    if train_model:

        # discretize forward returns into classes
        raw_df.dropna(subset=[y_col], inplace=True)
        raw_df.loc[:, y_col] = discret_rets(raw_df[y_col], cut_range, fwd_ret_labels)
        raw_df.dropna(subset=[y_col], inplace=True) # no nas in y_col
        print(sample_wgts(raw_df[y_col]))
        raw_df.loc[:, y_col] = raw_df[y_col].astype(str) # class as string

        if imputer_on: raw_df.loc[:, X_cols] = imputer.fit_transform(raw_df[X_cols])
        else: raw_df.dropna(inplace=True)

        X, y = raw_df.drop(columns=y_col), raw_df[y_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_sz, random_state=42)
        os.makedirs(ml_path, exist_ok=True)
        np.save(ml_path + train_cols, X_train.columns) # save feature order
    else:
        pred_X = raw_df.iloc[-pred_batch:,:].dropna(axis=0)

    [print(x.shape) for x in (pred_X, X_train, X_test, y_train, y_test) if x is not None]
    return pred_X, X_train, X_test, y_train, y_test

def train_ds(context):

    print('Benchmark: {}, Y: {}, Include: {}, invert: {}, include price: {}'.format(
    bench, y_col, include, invert, incl_price))

    context['train_model'] = True
    ml_path = context['ml_path']
    grid_search = context['grid_search']
    verbose = context['verbose']

    # create and pre-process datasets
    _, df_large = create_ds(px_close, context)
    pred_X, X_train, X_test, y_train, y_test = pre_process_ds(df_large, context)

    # RandomForestClassifier
    features = X_train.shape[1]
    best_params = { # best from GridSearch
        'n_estimators': 50,
        'max_features': features // 4,
        'max_depth': 30,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 2,
        'n_jobs': -1}
    if grid_search:
        print('GridSearchCV for RandomForestClassifier')
        param_grid = {
            'n_estimators': [50],
            'max_features': [features // x for x in range(1,10,1)],
            'max_depth': 30,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': np.arange(0, 5, 1),
        }
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
        'solver': 'sgd',
        'max_iter': 2000,
        'activation': 'relu',
        'alpha': 0.01,
        'hidden_layer_sizes': (features // 4),
        'learning_rate': 'adaptive',
        'random_state': 0, }
    if grid_search:
        print('GridSearchCV for MLPClassifier')
        param_grid = {
            'solver': ['sgd'], # ['lbfgs', 'sgd', 'adam']
            'max_iter': [2000], # [400, 1000, 2000]
            'activation': ['relu'], # ['logistic', 'tanh', 'relu']
            'alpha': [10.0 ** -2], # 10.0 ** -np.arange(2, 5, 1)
            'learning_rate' : ['adaptive'], # ['constant', 'adaptive']
            'hidden_layer_sizes': [features // 4], #[features // x for x in range(2,10,2)]
            'random_state': np.arange(0, 5, 1)
        }
        clf = GridSearchCV(MLPClassifier(random_state=42), param_grid, n_jobs=-1, cv=5,
                          iid=True, verbose=verbose)
        clf.fit(X_train, y_train)
        if verbose: mu.print_cv_results(
            clf, (X_train, X_test, y_train, y_test),
            feat_imp=False, top=20)
        best_params = clf.best_params_

    clf2 = MLPClassifier(**best_params)
    clf2.fit(X_train, y_train)
    print('MLPClassifier scores Train {}, Test {}'.format(
    clf2.score(X_train, y_train), clf2.score(X_test, y_test)))

    best_params = { # best from GridSearch
        'n_estimators': 50,
        'max_features': features // 2,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 2,
        'n_jobs': -1}
    # ExtraTreesClassifier
    clf3 = ExtraTreesClassifier(**best_params)
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
    train_cols = context['trained_cols']

    ds_idx, df_large = create_ds(px_close, context)
    pred_X, _, _, _, _ = pre_process_ds(df_large, context)
    print('pred_X.shape', pred_X.shape)

    # ensure prediction dataset is consistent with trained model
    trained_cols = np.load(ml_path + train_cols, allow_pickle=True) # save feature order
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
    s3_path = context['s3_path']
    idx_name = 'index' if bench_df.index.name is None else bench_df.index.name
    s3_df = bench_df.reset_index(drop=False)
    rename_col(s3_df, idx_name, 'pred_date')
    csv_store(s3_df, s3_path, csv_ext.format(tgt_date[0]))

    return bench_df

#context/config for training and prediction
context = {
    'portion': 100e-2,
    'ml_path': './ML/',
    'tmp_path': './tmp/',
    'trained_cols': 'macro_train_cols.npy',
    's3_path': 'recommend/macro_ML/',
    'px_close': 'universe-px-ds',
    'test_size': .20,
    'predict_batch': 252,
    'load_ds': True,
    'grid_search': False,
    'fill': 'ffill',
    'impute': True,
    'scale': True,
    'verbose': 1
}

px_close = load_px_close(
    context['tmp_path'],
    context['px_close'],
    context['load_ds'])[include].drop_duplicates()
# px_close = get_mults_pricing(include, freq, verbose=context['verbose']);
# px_close.drop_duplicates(inplace=True)
print('px_close.shape', px_close.shape)

dates = read_dates('quote')
tgt_date = [dates[-1]] # last date saved in S3

if __name__ == '__main__':
    hook = sys.argv[1]
    if hook == 'train':
        print('Training model using:', context)
        train_ds(context)
    elif hook == 'predict':
        print('Predicting model using:', context)
        pred_df = predict_ds(context)
        print(pred_df.tail(5).round(3).T)
    else: print('Invalid option, please try: train or predict')
