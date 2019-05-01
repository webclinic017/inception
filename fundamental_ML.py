# imports
import os
from utils.basic_utils import *
from utils.pricing import load_px_close, discret_rets, sample_wgts, dummy_col, rename_col
from utils.fundamental import pipe_transform_df, chain_divide, chain_scale, strips
from utils.fundamental import chain_outlier, chain_post_drop, chain_wide_transform
from utils.fundamental import chain_eps_estimates, chain_eps_revisions, chain_rec_trend
from utils.fundamental import load_append_ds, get_daily_ts, numeric_cols, filter_cols
from utils.ml_utils import show_fi, print_cv_results

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.validation import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.metrics import precision_score, roc_auc_score

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

# MODEL SPECIFIC FUNCTIONS
def create_ds(context):
    print('create_ds')
    # context variables
    key = context['key']
    load_ds = context['load_ds']
    load_dates = context['load_dates']
    tmp_path = context['tmp_path']
    tickers = context['tickers']

    # Load or append missing data to local dataset
    fname = tmp_path + key
    if load_ds & os.path.isfile(fname):
        daily_df = pd.read_parquet(fname)
        # compare and load missing dates
        missing_dates = list(
            set(daily_df.index.unique().date.astype(str))\
                .symmetric_difference(load_dates))
        if len(missing_dates) > 0: # retrieve missing dates
            append_df = get_daily_ts(key, ds_dict, missing_dates)
            daily_df = pd.concat([daily_df, append_df], axis=0) # append to daily
            # daily_df.drop_duplicates(inplace=True)
            daily_df.to_parquet(tmp_path + key) # and persist to drive for next time
    else:
        # file does not exist, retrieves all dates
        daily_df = get_daily_ts(key, ds_dict, load_dates)
        num_cols = excl(daily_df.columns, ['symbol', 'period'])
        daily_df.loc[:, num_cols] = daily_df[num_cols].astype(np.float32)
        # Make index a flat date, easier to index save down to drive if refresh pricing
        os.makedirs(tmp_path, exist_ok=True)
        # daily_df.drop_duplicates(inplace=True)
        daily_df.to_parquet(fname)

    daily_df.index.name = ds_dict[key]['index']
    daily_df.index = daily_df.index.date

    return daily_df

def pre_process_ds(daily_df, context):

    print('pre_process_ds')
    # context variables
    key = context['key']
    key_dict = context['ds_dict']
    fn_pipe = context['fn_pipeline']
    train_model = context['train_model']
    df_index = key_dict['index']
    (ml_path, _) = context['ml_path']
    trained_cols = context['trained_cols']
    impute_on, scale_on = key_dict = context['impute'], context['scale']

    # pre-process entire datasets in one shot or by company
    processed_df = pipe_transform_df(daily_df, key, fn_pipe, context)
    print('processed_df.shape', processed_df.shape)
    # processed_df.describe(include='all', percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).T

    scaler = MinMaxScaler()
    X_num_cols = excl(numeric_cols(processed_df), [y_col])
    processed_df.loc[:, X_num_cols] = processed_df[X_num_cols].replace([np.inf, -np.inf, 'NaN', np.nan], 0)
    if scale_on: processed_df.loc[:, X_num_cols] = scaler.fit_transform(processed_df[X_num_cols])

    ## add sectors columns
    # processed_df.loc[:, 'sector'] = processed_df.reset_index()['symbol'].map(profile.sector.to_dict()).values
    # processed_df.dropna(subset=['sector'], inplace=True)
    # for adding sectors, messy: needs improvement / refactoring
    # processed_df.describe(include='all').T
    ## turn categorical fields into one-hot encoder dummies
    # categ_dummies = {
    #     'day_quote': ['sector'],
    #     'key_statistics': ['sector'],
    #     'fin_data': [ 'sector'], # 'financialCurrency', 'recommendationKey',
    #     'eps_estimates': ['sector'],
    #     'eps_revisions': ['sector'],
    #     'eps_trend': ['sector'],
    #     'rec_trend': ['sector'],
    #     'net_purchase': ['sector'],
    # }
    # if key in categ_dummies:
    #     for col in categ_dummies[key]: processed_df = dummy_col(processed_df, col, shorten=True)

    X_pred = X_train = X_test = y_train = y_test = None
    if processed_df.index.is_mixed(): processed_df.reset_index(level=1, drop=False, inplace=True)

    if train_model:

        processed_df.index = pd.to_datetime(processed_df.index)
        print('processed_df.columns', processed_df.columns)

        # reshapes to include symbol in index in additional to date
        Y_df = Y.loc[processed_df.index.unique().sortlevel()[0], :]
        Y_df = Y_df.stack().to_frame().rename(columns={0: y_col})
        Y_df.index.set_names([df_index, 'symbol'], inplace=True)
        print('Y_df.shape', Y_df.shape)

        # re-index processed df on storeDate and symbol to have similar indices
        processed_df.index.set_names([df_index], inplace=True)
        processed_df.reset_index(inplace=True)
        processed_df.set_index([df_index, 'symbol'], inplace=True)
        # processed_df.reset_index(level=1) # to get back to date only, if needed
        print('processed_df.shape', processed_df.shape)

        # add Y values to processed df fast without having to loop
        processed_df.loc[:, y_col] = Y_df.loc[processed_df.index, y_col]

        # discretize Y-variable
        processed_df.dropna(subset=[y_col], inplace=True)
        processed_df[y_col] = discret_rets(processed_df[y_col], cut_range, fwd_ret_labels)
        print(sample_wgts(processed_df[y_col]))

        processed_df.dropna(subset=[y_col], inplace=True)
        processed_df.loc[:, y_col] = processed_df[y_col].astype(str)
        processed_df.reset_index(drop=True, inplace=True)

        X, y = processed_df.drop(columns=y_col), processed_df[y_col]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            break # just one split
        np.save(ml_path + trained_cols.format(key), X_train.columns) # save feature order
        print('Saved: ', ml_path + trained_cols.format(key))
    else:
        # feature for last date, pending to implement more flexibility
#         processed_df.reset_index(drop=True, inplace=True)
        X_pred = processed_df.loc[processed_df.index[-1], :]

    [print(x.shape) for x in (X_pred, X_train, X_test, y_train, y_test) if x is not None]

    return X_pred, X_train, X_test, y_train, y_test

def train_ds(context):
    print('train_ds')
    context['train_model'] = True
    grid_search = context['grid_search']
    verbose = context['verbose']
    (path, model_name) = context['ml_path']

    daily_df = create_ds(context)
    print(daily_df.info(verbose=False))

    X_pred, X_train, X_test, y_train, y_test = pre_process_ds(daily_df, context)
    features = X_train.shape[1]

    # RandomForestClassifier
    best_params = {
        'n_estimators': 50, 'max_features': features, 'max_depth': 30,
        'min_samples_split': 2,
        'random_state': 1, 'n_jobs': -1}
    if grid_search:
        print('GridSearchCV for RandomForestClassifier')
        param_grid = {
            'n_estimators': [50],
            'max_features': ['sqrt', 'log2', features // 2, features // 3,],
            'max_depth': [30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 5, 10],
            'random_state': np.arange(0, 3, 1),}
        clf = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid, n_jobs=-1, cv=5, iid=True, verbose=verbose)
        clf.fit(X_train, y_train)
        print_cv_results(clf, (X_train, X_test, y_train, y_test), feat_imp=True, top=20)
        best_params = clf.best_params_
    clf1 = RandomForestClassifier(**best_params)
    clf1.fit(X_train, y_train)
    print('RandomForestClassifier scores: Train {}, Test {}'.format(
        clf1.score(X_train, y_train),
        clf1.score(X_test, y_test)))

    # ExtraTreesClassifier
    best_params = {
    'n_estimators': 50, 'max_depth': 30,
    'min_samples_split': 2, 'min_samples_leaf': 1,
    'random_state': None, 'n_jobs': -1}
    clf2 = ExtraTreesClassifier(**best_params)
    clf2.fit(X_train, y_train)
    print('ExtraTreesClassifier scores: Train {}, Test {}'.format(
        clf2.score(X_train, y_train),
        clf2.score(X_test, y_test)))

    for vote in ['hard', 'soft']:
        eclf = VotingClassifier(estimators=[('rf', clf1), ('et', clf2)], voting=vote)
        clf = eclf.fit(X_train, y_train)
        print('VotingClassifier scores Train {}, Test {}'.format(
            clf.score(X_train, y_train),
            clf.score(X_test, y_test)))
        os.makedirs(path, exist_ok=True)
        fname = path + model_name.format(key, vote)
        joblib.dump(clf, fname)
        print('Saved ', fname)

def predict_ds(context):
    print('predict_ds')
    context['load_ds'] = True
    context['train_model'] = False
    (path, model_name) = context['ml_path']
    verbose = context['verbose']
    trained_cols = context['trained_cols']

    daily_df = create_ds(context)
    pred_X, _, _, _, _ = pre_process_ds(daily_df, context)
    print('predict_ds')
    print('pred_X.shape', pred_X.shape)

    # ensure prediction dataset is consistent with trained model
    trained_cols = np.load(path + trained_cols.format(key)) # save feature order
    missing_cols = [x for x in trained_cols if x not in pred_X.columns]
    pred_X = pd.concat([pred_X, pd.DataFrame(columns=missing_cols)], axis=1)
    pred_X[missing_cols] = 0
    pred_X = pred_X[list(trained_cols) + ['symbol']]

    pred_df = pd.DataFrame()
    pred_df['symbol'] = pred_X.symbol
    for vote in ['hard', 'soft']:
        fname = path + model_name.format(key, vote)
        clf = joblib.load(fname) # load latest models
        print('Loaded', fname)
        preds = clf.predict(pred_X.iloc[:, :-1])
        # preds = np.where(preds == 'nan', 'neutral', preds) #replace nan
        pred_class = np.array([fwd_ret_labels.index(x) for x in preds])
        pred_df[f'{vote}_pred_class'] = pred_class
        pred_df[f'{vote}_pred_label'] = preds
        if vote == 'soft':
            probs = clf.predict_proba(pred_X.iloc[:, :-1])
            pred_prob = np.argmax(probs, axis=1)
            pred_df[f'{vote}_confidence'] = [x[np.argmax(x)] for x in probs] # higest prob
            prob_df = pd.DataFrame(probs, index=pred_df.index, columns=clf.classes_)
            pred_df = pd.concat([pred_df, prob_df[fwd_ret_labels]], axis=1)

    # store in S3
    s3_path = context['s3_path']
    idx_name = 'index' if pred_df.index.name is None else pred_df.index.name
    s3_df = pred_df.reset_index(drop=False)
    rename_col(s3_df, idx_name, 'pred_date')
    csv_store(s3_df, s3_path, csv_ext.format(tgt_date[0]))

    return pred_df

# feature mapping for different datasets
ds_dict = {
    'fin_data': {
        'path': 'summary-categories/financialData/',
        'index': 'storeDate',
        'features': [
            'numberOfAnalystOpinions', 'currentPrice', 'revenuePerShare', 'totalCashPerShare',
            'currentRatio', 'debtToEquity', 'earningsGrowth', 'ebitda',
            'ebitdaMargins', 'freeCashflow', 'grossMargins',
            'grossProfits', 'operatingCashflow', 'operatingMargins', 'profitMargins',
            'quickRatio', 'recommendationMean',
            'returnOnAssets', 'returnOnEquity', 'revenueGrowth',
            'targetHighPrice', 'targetLowPrice', 'targetMeanPrice',
            'targetMedianPrice', 'totalCash', 'totalDebt', 'totalRevenue',
#             'financialCurrency', 'recommendationKey',
            'symbol', ],
        'scale': [
            'freeCashflow', 'operatingCashflow', 'ebitda',
            'totalCash', 'totalDebt', 'totalRevenue', 'grossProfits', ],
        'divide': ('currentPrice',
                   ['revenuePerShare', 'totalCashPerShare',
                    'targetLowPrice', 'targetMeanPrice',
                    'targetMedianPrice', 'targetHighPrice', ]),
        'outlier': 'quantile',
        'post_drop': ['numberOfAnalystOpinions'],
    },
    'key_statistics': {
        'path': 'summary-categories/defaultKeyStatistics/',
        'index': 'storeDate',
        'features': [
            'beta', 'earningsQuarterlyGrowth',
            'enterpriseToEbitda', 'enterpriseToRevenue', 'enterpriseValue', 'forwardPE',
            'netIncomeToCommon', 'pegRatio', 'priceToBook', 'profitMargins',
            'shortPercentOfFloat', 'shortRatio', 'heldPercentInsiders',
            'heldPercentInstitutions', 'symbol', ],
        'scale': ['enterpriseValue', 'netIncomeToCommon', ],
        'outlier': 'quantile',
    },
    'eps_trend': {
        'path': 'summary-categories/epsTrend/',
        'index': 'storeDate',
        'periods': ['0y', '+1y', '+5y', '-5y'],
        'features': [
            'period', 'growth',
            'current', '7daysAgo', '30daysAgo', '60daysAgo', '90daysAgo',
            'symbol', ],
        'pivot_cols': ['growth', 'current', '7daysAgo', '30daysAgo', '60daysAgo', '90daysAgo'],
        'outlier': 'quantile',
    },
    'eps_estimates': {
        'path': 'summary-categories/earningsEstimate/',
        'index': 'storeDate',
        'periods': ['0y', '+1y', '+5y', '-5y'],
        'features': [
            'period', 'growth', 'avg', 'low', 'high', 'symbol', ],
        'pivot_cols': ['growth', 'avg', 'low', 'high'],
        'outlier': 'quantile',
    },
    'day_quote': {
        'path': 'quote/csv/',
        'index': 'storeDate',
        'features': [
            'regularMarketChangePercent',
            'averageDailyVolume10Day', 'averageDailyVolume3Month', 'regularMarketVolume',
            'fiftyDayAverageChangePercent', 'twoHundredDayAverageChangePercent',
            'fiftyTwoWeekHighChangePercent', 'fiftyTwoWeekLowChangePercent',
            'forwardPE', 'trailingPE', 'priceToBook', 'marketCap', 'symbol', ],
        'scale': ['marketCap', ],
        'divide': ('regularMarketVolume', ['averageDailyVolume10Day', 'averageDailyVolume3Month']),
        'outlier': 4,
    },
    'eps_revisions': {
        'path': 'summary-categories/epsRevisions/',
        'index': 'storeDate',
        'periods': ['0y', '+1y', '+5y', '-5y'],
        'features': [
            'period', 'growth', 'upLast7days', 'upLast30days', 'downLast30days',
            'symbol', ],
        'pivot_cols': ['growth', 'upLast7days', 'upLast30days', 'downLast30days'],
        'outlier': 3,
    },
    'spy_trend': {
        'path': 'summary-categories/indexTrend/',
        'index': 'storeDate',
        'features': [
            '0q', '+1q', '0y', '+1y', '+5y', '-5y',
            'peRatio', 'pegRatio', 'symbol', ]
    },
    'net_purchase': {
        'path': 'summary-categories/netSharePurchaseActivity/',
        'index': 'storeDate',
        'features': [
            'netPercentInsiderShares', 'buyPercentInsiderShares', 'sellPercentInsiderShares',
            'symbol', ]
    },
    'rec_trend': {
        'path': 'summary-categories/recommendationTrend/',
        'index': 'storeDate',
        'periods': ['-1m', '-2m'],
        'features': [
            'period', 'strongBuy', 'buy', 'hold', 'sell', 'strongSell',
            'symbol', ],
        'pivot_cols': ['strongBuy', 'buy', 'hold', 'sell', 'strongSell'],
        'outlier': 10,
    },
}

# pre-processing pipeline
fn_pipeline = {
    'fin_data': [chain_scale, chain_divide, chain_post_drop, chain_outlier],
    'key_statistics': [chain_scale, chain_outlier],
    'eps_trend': [chain_wide_transform, chain_eps_estimates, chain_outlier],
    'eps_estimates': [chain_wide_transform, chain_eps_estimates, chain_outlier],
    'day_quote': [chain_divide, chain_scale, chain_outlier],
#     'eps_revisions': [chain_wide_transform, chain_outlier],
#     'spy_trend':[lambda x, y: x],
#     'net_purchase':[lambda x, y: x],
#     'rec_trend': [chain_wide_transform, chain_rec_trend, chain_outlier],
}

# environment variables
cuts = { '1d': [-1, -0.1, -.02, .02, .1, 1.] }
cut_range = cuts['1d']
fwd_ret_labels = ["bear", "short", "neutral", "long", "bull"]
look_ahead, look_back = 20, 252
y_col = 'fwdReturn' + f'{look_ahead}'

if __name__ == '__main__':
    key = sys.argv[1]
    hook = sys.argv[2]

    load_dates = read_dates(ds_dict[key]['path'], '.csv')[-look_back:]
    context = {
        'key': key,
        'pre': key.split('_')[0], # append preffix,
        'fn_pipeline': fn_pipeline,
        'ds_dict': ds_dict[key],
        'ml_path': ('./ML/', 'fdmn_ML-{}_{}.pkl'),
        'tmp_path': './tmp/',
        'ds_append': 'fund-ml-processed-',
        'px_close': 'universe-px-ds',
        'trained_cols': ('fdmn-{}_train_cols.npy'),
        's3_path': f'recommend/fdmn_ML-{key}/',
        'load_ds': True,
        'grid_search': False,
        'impute': False,
        'scale': False,
        'verbose': 0,
        # 'nbr_anr': stacked_anr,
    }

    symbols_list = config['companies']
    px_close = load_px_close(
    context['tmp_path'],
    context['px_close'],
    context['load_ds'])[symbols_list].drop_duplicates()
    print('px_close.info()', px_close.info())

    Y = px_close.pct_change(look_ahead).shift(-look_ahead)
    Y = Y[~(Y.isna().all(1))]

    quote_dates = read_dates('quote')
    tgt_date = quote_dates[-1:] # last quote saved in S3

    quotes = load_csvs('quote_consol', tgt_date) # metrics for last day
    profile = load_csvs('summary_detail', ['assetProfile']) # descriptive items
    quotes.set_index('symbol', drop=False, inplace=True)
    profile.set_index('symbol', drop=False, inplace=True)

    stacked_px = px_close.stack().to_frame().rename(columns={0: 'close'}) # stack date + symbol
    stacked_px.index.set_names(['storeDate', 'symbol'], inplace=True) # reindex

    # load data for a given date range, for training models
    load_dates = read_dates(ds_dict[key]['path'], '.csv')[-look_back:]

    # add additional items to the context
    context['tickers'] = symbols_list
    context['load_dates'] = load_dates
    context['close_px'] = stacked_px

    if hook == 'train':
        print('Training {} using:'.format(key))
        # print('Context: ', context)
        train_ds(context)
    elif hook == 'predict':
        print('Predicting {} using:'.format(key))
        # print('Context: ', context)
        pred_df = predict_ds(context)
        print(pred_df.tail(5).round(3).T)
    else: print('Invalid option, please try: train or predict')
