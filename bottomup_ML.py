# imports
import os
from matplotlib import pyplot as plt
from utils.basic_utils import *
from utils.pricing import load_px_close, discret_rets, sample_wgts
from utils.pricing import dummy_col, rename_col, px_fwd_rets, px_mom_feats, px_mom_co_feats_light
from utils.pricing import eq_wgt_indices, to_index_form, get_symbol_pricing
from utils.fundamental import pipe_transform_df, chain_divide, chain_scale
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
    # context variables
    key = context['key']
    load_ds = context['load_ds']
    load_dates = context['load_dates']
    tmp_path = context['tmp_path']

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

def create_pre_process_ds(context):

    # join all datasets
    super_list = []
    for key in ('fin_data', 'key_statistics', 'day_quote', 'eps_trend', 'eps_estimates'):
        print(f'adding {key}')
        context['key'] = key
        context['pre'] = key.split('_')[0] # append preffix
        context['ds_dict'] = ds_dict[key]
        load_dates = read_dates(ds_dict[key]['path'], '.csv')
        context['load_dates'] = load_dates
        df = create_ds(context)
        df = df.loc[df.symbol.isin(symbols_list),:]
        processed_df = pipe_transform_df(df, key, fn_pipeline, context)
        if key in ('fin_data', 'key_statistics', 'day_quote'):
            processed_df.index.name = 'storeDate'
            processed_df = processed_df.reset_index().set_index(['storeDate', 'symbol'])
        super_list.append(processed_df.drop_duplicates())
    processed_df = pd.concat(super_list, axis=1)
    print(f'processed_df.shape {processed_df.shape}')

    sectors = profile.loc[profile.symbol.isin(symbols_list)].sector.unique()
    industries = profile.loc[profile.symbol.isin(symbols_list)].industry.unique()
    bench = '^GSPC'
    print(f'Sectors: {sectors.shape[0]}, Industries: {industries.shape[0]}')

    indices_df = pd.concat(
        [eq_wgt_indices(profile, px_close, 'sector', sectors, subset=symbols_list),
        eq_wgt_indices(profile, px_close, 'industry', industries, subset=symbols_list),
        to_index_form(get_symbol_pricing(bench)['close'], bench)],
        axis=1).drop_duplicates()

    # create price momentum features
    tmp_path = context['tmp_path']
    px_mom_fname = 'px_mom_feat_light'

    if os.path.isfile(tmp_path + px_mom_fname):
        px_mom_df = pd.read_parquet(tmp_path + px_mom_fname)
    else:
        super_list = []
        tickers = context['tickers']
        for i, ticker in enumerate(tickers):
            try:
                close = px_close[ticker].dropna()
                ft_df = px_mom_feats(close, ticker, incl_name=False)
                ft_df = ft_df.loc[processed_df.index.levels[0].sort_values(), :]
                top_groups = tuple([bench, profile.loc[ticker, 'sector']])
                co = px_mom_co_feats_light(close, indices_df, top_groups)
                ft_df = pd.concat([ft_df, co.loc[ft_df.index, :]], axis=1)
                super_list.append(ft_df.copy())
            except Exception as e:
                print("Exception: {0} {1}".format(ticker, e))
        px_mom_df = pd.concat(super_list, axis=0)
        px_mom_df = px_mom_df.reset_index().set_index(['storeDate', 'symbol']).sort_index().dropna()
        os.makedirs(tmp_path, exist_ok=True)
        px_mom_df.to_parquet(tmp_path + px_mom_fname)

    joined_df = pd.concat([processed_df, px_mom_df], join='inner', axis=1)
    print(f'joined_df.shape {joined_df.shape}')

    # basic impute and scaling
    scale_on = context['scale']
    scaler = StandardScaler()
    num_cols = numeric_cols(joined_df)
    joined_df.loc[:, num_cols] = joined_df[num_cols].replace([np.inf, -np.inf, np.nan], 0)
    if scale_on: joined_df.loc[:, num_cols] = scaler.fit_transform(joined_df[num_cols])

    # add categoricals
    joined_df = dummy_col(joined_df, 'sector', shorten=True)

    return joined_df.reset_index('symbol')

def train_ds(context):

    context['train_model'] = True
    grid_search = context['grid_search']
    verbose = context['verbose']
    (ml_path, model_name) = context['ml_path']
    train_model = context['train_model']
    trained_cols = context['trained_cols']
    look_ahead, smooth_window = context['look_ahead'], context['smooth_window']

    joined_df = create_pre_process_ds(context)

    # calculation of forward returns

    Y = px_close.pct_change(look_ahead).shift(-look_ahead)
    Y = Y.rolling(smooth_window).mean() # smooth by the same length
    Y = Y[~(Y.isna().all(1))]
    Y = Y.loc[joined_df.index.unique(), :]

    # reshapes to include symbol in index in additional to date
    Y_df = Y.loc[joined_df.index.unique().sortlevel()[0], :]
    Y_df = Y_df.stack().to_frame().rename(columns={0: y_col})
    # somwhat repetitive with steps above but performs faster
    Y_df.index.set_names(['storeDate', 'symbol'], inplace=True)
    print('Y_df.shape', Y_df.shape)

    # re-index processed df on storeDate and symbol to have similar indices
    joined_df.index.set_names('storeDate', inplace=True)
    joined_df.reset_index(inplace=True)
    joined_df.set_index(['storeDate', 'symbol'], inplace=True)
    print('joined_df.shape', joined_df.shape)

    # add Y values to processed df fast without having to loop
    joined_df.loc[:, y_col] = Y_df.loc[joined_df.index, y_col]

    # discretize Y-variable
    joined_df.dropna(subset=[y_col], inplace=True)
    joined_df[y_col] = discret_rets(joined_df[y_col], cut_range, fwd_ret_labels)
    print(sample_wgts(joined_df[y_col]))

    joined_df.dropna(subset=[y_col], inplace=True)
    joined_df.loc[:, y_col] = joined_df[y_col].astype(str)
    joined_df.reset_index(drop=True, inplace=True)

    X, y = joined_df.drop(columns=y_col), joined_df[y_col]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        break # just one split

    np.save(ml_path + trained_cols, X_train.columns) # save feature order
    print(f'X_train.shape {X_train.shape}')
    print(f'X_train columns {X_train.columns}')
    print('Saved: ', ml_path + trained_cols)

    # RandomForestClassifier
    clf1 = RandomForestClassifier()
#     clf1.fit(X_train, y_train)
#     print(f'Train {clf1.score(X_train, y_train)}, Test {clf1.score(X_test, y_test)}')

    # MLPClassifier
    neurons = X_train.shape[1] * 2
    mlp_params = {
        'solver': 'adam', 'max_iter': 200, #reduced from 600 for testing
        'hidden_layer_sizes': (neurons, neurons, neurons, neurons, neurons,),
        'n_iter_no_change': 10, 'verbose': True, 'random_state': None, }
    clf2 = MLPClassifier(**mlp_params)
#     clf2.fit(X_train, y_train)
#     print(f'Train {clf2.score(X_train, y_train)}, Test {clf2.score(X_test, y_test)}')

    # ExtraTreesClassifier
    clf3 = ExtraTreesClassifier()
#     clf3.fit(X_train, y_train)
#     print(f'Train {clf3.score(X_train, y_train)}, Test {clf3.score(X_test, y_test)}')

    for vote in ['hard', 'soft']:
        eclf = VotingClassifier(estimators=[('rf', clf1), ('mlp', clf2), ('et', clf3)], voting=vote)
        clf = eclf.fit(X_train, y_train)
        print(f'VotingClassifier Train {clf.score(X_train, y_train)}, Test {clf.score(X_test, y_test)}')
        os.makedirs(ml_path, exist_ok=True)
        fname = ml_path + model_name.format(vote)
        joblib.dump(clf, fname)
        print('Saved ', fname)#     return processed_df

def predict_ds(context):

    print('predict_ds')
    context['load_ds'] = True
    context['train_model'] = False
    (ml_path, model_name) = context['ml_path']
    verbose = context['verbose']
    trained_cols_fname = context['trained_cols']

    joined_df = create_pre_process_ds(context)
    pred_X = joined_df.loc[joined_df.index[-1], :]

    # ensure prediction dataset is consistent with trained model
    train_cols = np.load(ml_path + trained_cols_fname) # save feature order
    missing_cols = [x for x in train_cols if x not in pred_X.columns]
    if len(missing_cols):
        pred_X = pd.concat([pred_X, pd.DataFrame(columns=missing_cols)], axis=1)
        pred_X[missing_cols] = 0
    print('pred_X.shape', pred_X.shape)

    sorted_cols = list(np.append(train_cols, ['symbol']))
    print('pred_X.shape', pred_X[sorted_cols].shape)

    pred_df = pd.DataFrame()
    pred_df['symbol'] = pred_X.symbol
    for vote in ['hard', 'soft']:
        fname = ml_path + model_name.format(vote)
        clf = joblib.load(fname) # load latest models
        print('Loaded', fname)
        preds = clf.predict(pred_X[sorted_cols].iloc[:, :-1])
        # preds = np.where(preds == 'nan', 'neutral', preds) #replace nan
        pred_class = np.array([fwd_ret_labels.index(x) for x in preds])
        pred_df[f'{vote}_pred_class'] = pred_class
        pred_df[f'{vote}_pred_label'] = preds
        if vote == 'soft':
            probs = clf.predict_proba(pred_X[sorted_cols].iloc[:, :-1])
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
            'enterpriseToEbitda', 'enterpriseToRevenue', 'enterpriseValue',
            'netIncomeToCommon', 'pegRatio',
            'shortPercentOfFloat', 'shortRatio', 'heldPercentInsiders',
            'heldPercentInstitutions', 'symbol', ],
        'scale': ['enterpriseValue', 'netIncomeToCommon', ],
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
        'features': ['period', 'avg', 'low', 'high', 'symbol', ],
        'pivot_cols': ['avg', 'low', 'high'],
        'outlier': 'quantile',
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
    'day_quote': [chain_divide, chain_scale, chain_outlier],
    'eps_trend': [chain_wide_transform, chain_eps_estimates, chain_outlier],
    'eps_estimates': [chain_wide_transform, chain_eps_estimates, chain_outlier],
#     'eps_revisions': [chain_wide_transform, chain_outlier],
#     'spy_trend':[lambda x, y: x],
#     'net_purchase':[lambda x, y: x],
#     'rec_trend': [chain_wide_transform, chain_rec_trend, chain_outlier],
}

# environment variables
cut_range = [-np.inf, -0.12, -.04, .04, .12, np.inf]
fwd_ret_labels = ["bear", "short", "neutral", "long", "bull"]


if __name__ == '__main__':

    symbols_list = config['companies']
    context = {
        'tickers': symbols_list,
        'fn_pipeline': fn_pipeline,
        'look_ahead': 20,
        'smooth_window': 10,
        'ml_path': ('./ML/', 'bottomup_ML_{}.pkl'),
        'tmp_path': './tmp/',
        'ds_append': 'fund-ml-processed-',
        'px_close': 'universe-px-ds',
        'trained_cols': ('bottomup-ML_train_cols.npy'),
        's3_path': f'recommend/bottomup_ML/',
        'load_ds': True,
        'grid_search': False,
        'impute': False,
        'scale': True,
        'verbose': 0,
        # 'nbr_anr': stacked_anr,
    }

    y_col = 'fwdReturn' + f'{context["look_ahead"]}'
    px_close = load_px_close(
        context['tmp_path'],
        context['px_close'],
        context['load_ds'])[symbols_list].drop_duplicates()
    print('px_close.info()', px_close.info())

    stacked_px = px_close.stack().to_frame().rename(columns={0: 'close'}) # stack date + symbol
    stacked_px.index.set_names(['storeDate', 'symbol'], inplace=True) # reindex
    context['close_px'] = stacked_px

    quote_dates = read_dates('quote')
    tgt_date = quote_dates[-1:] # last quote saved in S3

    quotes = load_csvs('quote_consol', tgt_date) # metrics for last day
    profile = load_csvs('summary_detail', ['assetProfile']) # descriptive items
    quotes.set_index('symbol', drop=False, inplace=True)
    profile.set_index('symbol', drop=False, inplace=True)

    hook = sys.argv[1]
    if hook == 'train':
        print('Training...')
        # print('Context: ', context)
        train_ds(context)
    elif hook == 'predict':
        print('Predicting...')
        # print('Context: ', context)
        pred_df = predict_ds(context)
        print(pred_df.tail(5).round(3).T)
    else: print('Invalid option, please try: train or predict')
