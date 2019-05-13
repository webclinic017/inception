# imports
import time, os, sys
from tqdm import tqdm

# from matplotlib import pyplot as plt
from utils.basic_utils import *
from utils.pricing import load_px_close, discret_rets, sample_wgts
from utils.pricing import dummy_col, rename_col, px_fwd_rets, px_mom_feats, px_mom_co_feats_light
from utils.pricing import eq_wgt_indices, to_index_form, get_symbol_pricing, get_return_intervals
from utils.fundamental import pipe_transform_df, chain_divide, chain_scale
from utils.fundamental import chain_outlier, chain_post_drop, chain_wide_transform
from utils.fundamental import chain_eps_estimates, chain_eps_revisions, chain_rec_trend
from utils.fundamental import load_append_ds, get_daily_ts, numeric_cols, filter_cols
from utils.fundamental import get_focus_tickers

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

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, Adamax, Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import CSVLogger

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

pd.options.display.float_format = '{:,.2f}'.format

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
bench = '^GSPC'
y_col = 'fwdReturn'
tickers = config['companies']

context = {
    'tickers': tickers,
    'fn_pipeline': fn_pipeline,
    'ml_path': '../ML/',
    'model_name': 'bottomup_TF.h5',
    'tmp_path': '../tmp/',
    'ds_name': 'co-bottomup-ds',
    'px_close': 'universe-px-ds',
    'trained_cols': 'bottomup_TF_train_cols.npy',
    'look_ahead': 20,
    'look_back': 252,
    'smooth_window': 10,
    'load_ds': True,
    'scale': True,
    'test_size': .05,
    'verbose': True,
    's3_path': f'recommend/bottomup_ML/',
    'verbose': 2,
    'units': 1000,
    'hidden_layers': 4,
    'max_iter': 400,
    'l2_reg': 0.5,
    'dropout': 0.5,
}

px_close = load_px_close(
    context['tmp_path'], context['px_close'], context['load_ds']).drop_duplicates()
print('px_close.info()', px_close.info())

stacked_px = px_close.stack().to_frame().rename(columns={0: 'close'}) # stack date + symbol
stacked_px.index.set_names(['storeDate', 'symbol'], inplace=True) # reindex
context['close_px'] = stacked_px

prices = px_close.dropna(subset=[bench])[tickers]
look_ahead = context['look_ahead']
# cut_range = get_return_intervals(prices, look_ahead, tresholds=[0.25, 0.75])
# hardcoded to narrow the range of recomendation in these limited dataset
cut_range = [ -np.inf, -0.13, -0.08,  0.1, 0.16, np.inf]
fwd_ret_labels = ["bear", "short", "neutral", "long", "bull"]
f'Return intervals {np.round(cut_range, 2)}'

# latest quotes, profile, and industries
dates = read_dates('quote')
tgt_date = dates[-1] # last date saved in S3
print(f'Target date: {tgt_date}')

quotes = load_csvs('quote_consol', [tgt_date])
quotes.set_index('symbol', drop=False, inplace=True)

profile = load_csvs('summary_detail', ['assetProfile'])
profile.set_index('symbol', drop=False, inplace=True)


# MODEL SPECIIFIC FUNCTIONS
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

def pre_process_ds(context):

    # join all datasets
    tickers = context['tickers']
    sectors = profile.loc[profile.symbol.isin(tickers)].sector.unique()
    industries = profile.loc[profile.symbol.isin(tickers)].industry.unique()
    print(f'Sectors: {sectors.shape[0]}, Industries: {industries.shape[0]}')

    indices_df = pd.concat(
        [eq_wgt_indices(profile, px_close, 'sector', sectors, subset=tickers),
        eq_wgt_indices(profile, px_close, 'industry', industries, subset=tickers),
        to_index_form(px_close[bench], bench)],
        axis=1).drop_duplicates()

    # generic price momentum statistics
    super_list = []
    for key in ('fin_data', 'key_statistics', 'day_quote', 'eps_trend', 'eps_estimates'):
        print(f'adding {key}')
        context['key'] = key
        context['pre'] = key.split('_')[0] # append preffix
        context['ds_dict'] = ds_dict[key]
        load_dates = read_dates(ds_dict[key]['path'], '.csv')
        context['load_dates'] = load_dates
        df = create_ds(context)
        df = df.loc[df.symbol.isin(tickers),:]
        processed_df = pipe_transform_df(df, key, fn_pipeline, context)
        if key in ('fin_data', 'key_statistics', 'day_quote'):
            processed_df.index.name = 'storeDate'
            processed_df = processed_df.reset_index().set_index(['storeDate', 'symbol'])
        super_list.append(processed_df.drop_duplicates())

    processed_df = pd.concat(super_list, axis=1)
    print(f'processed_df.shape {processed_df.shape}')

    # company specific statistics
    tmp_path = context['tmp_path']
    ds_name = context['ds_name']

    super_list = []
    for i, ticker in tqdm(enumerate(tickers)):
        try:
            close = px_close[ticker].dropna()
            ft_df = px_mom_feats(close, ticker, incl_name=False)
            if ticker in profile.symbol.unique():
                top_groups = tuple([bench, profile.loc[ticker, 'sector']])
                co = px_mom_co_feats_light(close, indices_df, top_groups)
                ft_df = pd.concat([ft_df, co.loc[ft_df.index, :]], axis=1)
                ft_df.index.name = 'storeDate'
                super_list.append(ft_df.copy())
            else: print(ticker, 'missing profile, skipping')
        except Exception as e:
            print("Exception: {0} {1}".format(ticker, e))

    px_mom_df = pd.concat(super_list, axis=0)

    px_mom_df = px_mom_df.reset_index().set_index(['storeDate', 'symbol']).sort_index().dropna()
    joined_df = pd.concat([processed_df, px_mom_df], join='inner', axis=1)
    print(f'joined_df.shape {joined_df.shape}')

    # basic scaling
    scale_on = context['scale']
    scaler = StandardScaler()
    num_cols = numeric_cols(joined_df)
    joined_df.loc[:, num_cols] = joined_df[num_cols].replace([np.inf, -np.inf, np.nan], 0)
    if scale_on: joined_df.loc[:, num_cols] = scaler.fit_transform(joined_df[num_cols])

    # add categoricals
    joined_df = dummy_col(joined_df, 'sector', shorten=True)

    return joined_df.reset_index('symbol')

def get_train_test_sets(context):

    verbose = context['verbose']
    ml_path, model_name = context['ml_path'], context['model_name']
    test_size = context['test_size']
    look_ahead, look_back, smooth_window = context['look_ahead'], context['look_back'], context['smooth_window']

    joined_df = pre_process_ds(context)

    # if we want to limit training set
    # index = joined_df.sort_index().index.unique()[-look_back:]
    # joined_df = joined_df.loc[index, :]
    # joined_df.shape

    # calculation of forward returns
    Y = px_close.loc[:, tickers].pct_change(look_ahead).shift(-look_ahead)
    Y = Y.rolling(smooth_window).mean() # smooth by the same length
    Y = Y[~(Y.isna().all(1))]
    Y = Y.loc[joined_df.index.unique(), :]

    # reshapes to include symbol in index in additional to date
    Y_df = Y.loc[joined_df.index.unique().sortlevel()[0], tickers]
    Y_df = Y_df.stack().to_frame().rename(columns={0: y_col})
    # somwhat repetitive with steps above but performs faster
    Y_df.index.set_names(['storeDate', 'symbol'], inplace=True)
    print('Y_df.shape', Y_df.shape)

    # re-index processed df on storeDate and symbol to have similar indices
    joined_df.index.set_names('storeDate', inplace=True)
    joined_df.set_index(['symbol'], append=True, inplace=True)
    print('joined_df.shape', joined_df.shape)

    # add Y values to processed df fast without having to loop
    joined_df.loc[:, y_col] = Y_df.loc[joined_df.index, y_col]

    # joined_df.loc[(slice(None), 'AAPL'), y_col].plot() # visualize smoothing
    # joined_df.groupby('symbol')[y_col].mean().sort_values() # rank long-term mean performance

    # discretize Y-variable
    joined_df.dropna(subset=[y_col], inplace=True)
    joined_df[y_col] = discret_rets(joined_df[y_col], cut_range, fwd_ret_labels)
    print('joined_df.shape', joined_df.shape)
    print(sample_wgts(joined_df[y_col]))

    joined_df.dropna(subset=[y_col], inplace=True)
    joined_df.loc[:, y_col] = joined_df[y_col].astype(str)

    days = len(joined_df.index.levels[0].unique())
    print(f'Training for {days} dates, {round(days/252, 1)} years')

    # joined_df.loc[(slice(None), 'TAL'), y_col].value_counts() # look at a specific security distribution
    train_df = joined_df.reset_index(drop=True)
    train_df.shape

    # create training and test sets
    X, y = train_df.drop(columns=y_col), train_df[y_col]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        break # just one split

    # skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    # for train_index, test_index in skf.split(X, y):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #     break

    return X_train, X_test, y_train, y_test

def train_ds(context):

    X_train, X_test, y_train, y_test = get_train_test_sets(context)

    # Keras Model
    units = context['units']
    max_iter = context['max_iter']
    l2_reg = context['l2_reg']
    dropout = context['dropout']
    trained_cols = context['trained_cols']

    y_train_oh = pd.get_dummies(y_train)[fwd_ret_labels]
    y_test_oh = pd.get_dummies(y_test)[fwd_ret_labels]

    # keras.regularizers.l2(l=0.001)

    model = Sequential()
    model.add(Dense(units, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(int(units/2), activation='relu'))
    model.add(Dense(len(pd.unique(y_train)), activation='softmax'))

    ml_path, model_name = context['ml_path'], context['model_name']
    fname = ml_path + model_name

    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
    checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
    csv_logger = CSVLogger('bottomup-train.log')

    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(
        X_train, y_train_oh, validation_data=(X_test, y_test_oh),
        epochs=max_iter, batch_size=200, callbacks=[es, checkpointer, csv_logger])

    score = model.evaluate(X_test, y_test_oh)
    print(f'Loss: {score[0]}, Accuracy: {score[1]}')

    # save training columns
    np.save(ml_path + trained_cols, X_train.columns) # save feature order
    print(f'X_train.shape {X_train.shape}, columns: {list(X_train.columns)}')
    print('Saved: ', ml_path + trained_cols)

    # save model to drive
    ml_path, model_name = context['ml_path'], context['model_name']
    fname = ml_path + model_name
    model.save(fname)
    print('Saved ', fname)

def predict_ds(context):

    ml_path = context['ml_path']
    model_name = context['model_name']
    trained_cols = context['trained_cols']

    joined_df = pre_process_ds(context)
    pred_X = joined_df.loc[joined_df.sort_index().index[-1], :]
    print('pred_X.shape', pred_X.shape)

    # ensure prediction dataset is consistent with trained model
    train_cols = np.load(ml_path + trained_cols, allow_pickle=True) # save feature order
    missing_cols = [x for x in train_cols if x not in pred_X.columns]
    if len(missing_cols):
        print(f'Warning missing columns: {missing_cols}')
        pred_X = pd.concat([pred_X, pd.DataFrame(columns=missing_cols)], axis=1)
        pred_X[missing_cols] = 0

    sorted_cols = list(np.append(train_cols, ['symbol']))
    print('pred_X.shape', pred_X[sorted_cols].shape)

    pred_df = pd.DataFrame()
    pred_df['symbol'] = pred_X.symbol

    # Load model
    fname = ml_path + model_name
    model = load_model(fname)
    print('Loaded', fname)

    preds = model.predict(pred_X[sorted_cols].iloc[:, :-1])
    preds_classes = model.predict_classes(pred_X[sorted_cols].iloc[:, :-1])

    pred_df['pred_class'] = preds_classes
    pred_df['pred_label'] = list(map(lambda x: fwd_ret_labels[x], preds_classes))
    probs = np.round(preds,3)
    pred_prob = np.argmax(probs, axis=1)
    pred_df['confidence'] = [x[np.argmax(x)] for x in probs] # higest prob
    prob_df = pd.DataFrame(probs, index=pred_df.index, columns=fwd_ret_labels)
    pred_df = pd.concat([pred_df, prob_df[fwd_ret_labels]], axis=1)
    pred_df.index.name = 'pred_date'

    # store in S3
    s3_path = context['s3_path']
    s3_df = pred_df.reset_index(drop=False)
    rename_col(s3_df, 'index', 'pred_date')
    csv_store(s3_df, s3_path, csv_ext.format(tgt_date))

    return pred_df


if __name__ == '__main__':
    hook = sys.argv[1]

    # Smaller subset for testing
    tgt_sectors = [
        'Technology',
        'Healthcare',
        'Industrials',
        'Basic Materials',
        'Consumer Cyclical',
        'Financial Services',
        'Consumer Defensive',
        'Real Estate',
        'Utilities',
        'Communication Services',
        'Energy',
    ]

    tickers = list(quotes.loc[quotes.quoteType == 'EQUITY', 'symbol'])
    context['tickers'] = tickers
    print(f'{len(tickers)} companies')

    if hook == 'train':
        # train with 50 random tickers, keep model small, same results
        print('Training...')
        train_ds(context)

    elif hook == 'predict':
        print('Predicting...')
        predict_ds(context)

    else: print('Invalid option, please try: train or predict')
