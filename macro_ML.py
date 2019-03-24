# imports
from utils.basic_utils import *
from utils.pricing import *
from utils.fundamental import *
from utils.imports import *
from utils.structured import *

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
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
print('All symbols: ', symbols_list)

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

f'Include: {include}, invert: {invert}, include price: {incl_price}\
Benchmark: {bench}, Y-var: {y_col}'

# utility functions
def create_ds(px_close):
    """ Create macro dataset and filtering index"""

    verbose = context['verbose']
    # average the return of the next periods
    # select only rows where Y variable is not null
    ds_idx = px_close.dropna(subset=[bench]).index
    Y = px_fwd_rets(px_close.loc[ds_idx, bench], bench, pred_fwd_windows).mean(axis=1)
    if verbose: print('Y.shape: ', Y.shape)

    df_large = pd.DataFrame()
    # rate transforms
    rate_ft_df = rate_feats(px_close[rateSL], rate_windows)
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
    df_large[y_col] = Y

    # drop NAs before discretizing
    df_large = df_large.loc[ds_idx, :]
    if verbose: print('df_large.shape: ', df_large.shape)

    return ds_idx, df_large

def pre_process_ds(df, context):

    verbose = context['verbose']
    portion = context['portion']

    train_model = context['train_model']
    ffill, imputer_on, scaler_on = \
        context['ffill'], context['impute'], context['scale']
    test_sz = context['test_size']
    pred_batch = context['predict_batch']
    imputer = SimpleImputer(
        missing_values=np.nan, strategy='median', copy=False)
    scaler = StandardScaler()
    X_cols = excl(df.columns, [y_col])

    # reduces the dataset in case is too large
    _, df_raw = train_test_split(df, test_size=portion, shuffle=False, )
    if verbose: print('Reduced df_raw.shape: ', df_raw.shape)

    df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    if ffill: df_raw.fillna(method='ffill', inplace=True)
    if scaler_on: df_raw.loc[:, X_cols] = scaler.fit_transform(df_raw[X_cols])

    pred_X = df_raw.iloc[-pred_batch:,:-1].copy() # how far back to predict
    X_train = X_test = y_train = y_test = None

    if train_model:
        if ffill: df_raw[X_cols].fillna(method='ffill', inplace=True)

        # for training, discretize forward returns into classes
        df_raw.dropna(subset=[y_col], inplace=True)
        df_raw[y_col] = discret_rets(df_raw[y_col], cut_range, fwd_ret_labels)
        df_raw[y_col] = df_raw[y_col].astype(str)

        y_col_dist = sample_wgts(df_raw[y_col], fwd_ret_labels)
        if verbose: print((y_col_dist[fwd_ret_labels]).round(3))

        # this seems unnecesary
        if imputer_on: df_raw.loc[:, X_cols] = imputer.fit_transform(df_raw[X_cols])
        else: df_raw[X_cols].dropna(inplace=True)

        X, y = df_raw.drop(columns=y_col), df_raw[y_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_sz, random_state=42)

    return pred_X, X_train, X_test, y_train, y_test

def train_ds(context):

    ml_path = context['ml_path']
    verbose = context['verbose']

    px_close = get_mults_pricing(include, freq, verbose=False);
    px_close.drop_duplicates(inplace=True)

    # create and pre-process datasets
    train_idx, df_large = create_ds(px_close)
    if verbose: print('df_large.shape: ', df_large.shape)
    pred_X, X_train, X_test, y_train, y_test = pre_process_ds(df_large, context)
    if verbose:
        for x in zip(('pred', 'X_train', 'y_train', 'X_test', 'y_test'),
        (pred_X, X_train, y_train, X_test, y_test)):
            print(x[0] + '.shape', x[1].shape)

    # PENDING: gridsearch models

    # RandomForestClassifier, best params from prior GridSearch
    rfc_params = {
        'max_features': 40, 'n_estimators': 100, 'random_state': 7}
    clf1 = RandomForestClassifier(**rfc_params, warm_start=True)
    clf1.fit(X_train, y_train)
    scores = clf1.score(X_train, y_train), clf1.score(X_test, y_test)

    # MLPClassifier, best params from prior GridSearch
    mlp_params = {
        'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': 65,
        'learning_rate': 'adaptive', 'max_iter': 200,
        'random_state': 3, 'solver': 'lbfgs'}
    clf2 = MLPClassifier(**mlp_params)
    clf2.fit(X_train, y_train)
    scores = clf2.score(X_train, y_train), clf2.score(X_test, y_test)

    # ExtraTreesClassifier
    clf3 = ExtraTreesClassifier(
        n_estimators=100, max_depth=None,
        min_samples_split=2, random_state=0)
    clf3.fit(X_train, y_train)
    scores = clf3.score(X_train, y_train), clf3.score(X_test, y_test)

    for vote in ['hard', 'soft']:
        eclf = VotingClassifier(
            estimators=[('rf', clf1), ('mlp', clf2), ('et', clf3)],
            voting=vote)
        clf = eclf.fit(X_train, y_train)
        scores = clf.score(X_train, y_train), clf.score(X_test, y_test)
        if verbose:
            print(clf)
            print(scores)

        os.makedirs(ml_path, exist_ok=True)
        fname = ml_path + f'macro_ML_{vote}.pkl'
        joblib.dump(clf, fname)
        print('Saved ', fname)

def predict_ds(context):
    ml_path = context['ml_path']
    verbose = context['verbose']

    px_close = get_mults_pricing(include, freq, verbose=False);
    px_close.drop_duplicates(inplace=True)

    ds_idx, df_large = create_ds(px_close)
    pred_X, _, _, _, _ = pre_process_ds(df_large, context)

    bench_df = px_close.loc[pred_X.index, bench].to_frame()
    bench_df.dropna(subset=[bench], inplace=True)

    # load latest models
    for vote in ['hard', 'soft']:
        fname = ml_path + f'macro_ML_{vote}.pkl'
        clf = joblib.load(fname)
        print('Loaded', fname)
        preds = clf.predict(pred_X)
#         pred_class = [x for x in map(fwd_ret_labels.index, preds)]
        pred_class = np.array([fwd_ret_labels.index(x) for x in preds])
        bench_df[f'{vote}_pred_class'] = pred_class
        bench_df[f'{vote}_pred_label'] = preds
        if vote == 'soft':
            probs = clf.predict_proba(pred_X)
            if verbose: print(clf.classes_)
            pred_prob = np.argmax(probs, axis=1)
#             bench_df[f'{vote}_high_prob_pred_class'] = pred_prob
            bench_df[f'{vote}_confidence'] = [x[np.argmax(x)] for x in probs] # higest prob
            prob_df = pd.DataFrame(probs, index=bench_df.index, columns=clf.classes_)
            bench_df = pd.concat([bench_df, prob_df[fwd_ret_labels]], axis=1)
        bench_df.dropna(subset=[bench], inplace=True)

    return bench_df

def visualize_predictions(pred_df):
    pre_class_cols = filter_cols(pred_df.columns, "pred_class")
    pred_df.loc[:,[bench] + pre_class_cols].plot(
        secondary_y=pre_class_cols, figsize=(15, 5));
    pred_df[fwd_ret_labels].plot.area(
            title='Prediction Probabilities',
            figsize=(15, 2), ylim=(0, 1), cmap='RdYlGn');
    f'Confidence Mean: {pred_df["soft_confidence"].mean().round(3)}, Median {pred_df["soft_confidence"].median().round(3)}'

# PENDING
def append_pricing(symbol, freq='1d', cols=None):
    """ appends most recent pricing to data on S3"""
    return appended_df

def pull_latest_px(tickers):
    """ get appended pricing from dataset """
    return px_close

#context/config for training and prediction
context = {
    'portion': 100e-2,
    'ffill': True,
    'impute': True,
    'scale': True,
    'test_size': .20,
    'predict_batch': 252,
    'ml_path': './ML/',
    'verbose': True}

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
        pred_df.tail(5).round(3).T
        s3_df = pred_df.reset_index(drop=False)
        rename_col(s3_df, 'index', 'pred_date')
        csv_store(s3_df, 'recommend/', 'macro_risk_ML.csv')
    else: print('Invalid option, please try: train or predict')
