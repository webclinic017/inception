from datetime import datetime, date

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from basic_utils import *

# lambdas
def date_lambda(x): return date.fromtimestamp(x)
def pd_datetime(x): return pd.to_datetime(x)
def datetime_lambda(x): return datetime.fromtimestamp(x) if x > 0 else 0
def time_delta_to_years(x): return x / 365
def divide_by_mean(x): return x / x.median()  # consider z-score
def cap_at_1q(x): return [max(y, 0) for y in x]
def z_score(x): return (x - x.mean()) / x.std()

# format / clean up functions
def clean_up_fmt(df):
    fmt_cols = [x for x in df.columns.tolist() if str.lower(x).find('fmt') > 0]
    raw_cols = [x for x in df.columns.tolist() if str.lower(x).find('raw') > 0]
    rndm_map = {x: x[:x.index('.')] for x in raw_cols}
    df.drop(fmt_cols, axis=1, inplace=True)
    df.rename(rndm_map, axis=1, inplace=True)
    return df

def cols_to_date(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(date_lambda)

def cols_to_bool(df, cols):
    for c in cols:
        df[c] = df[c].apply(lambda x: pd.to_numeric(x))

def divide_by(df, cols, tgt):
    cols.extend([tgt])
    res = (df[cols].T / df[cols][tgt]).T
    cols.remove(tgt)
    return res[cols]

# transforms and utilities
def merge_ds(fs_df, q_df):
    # data type conversions
    date_cols = ['regularMarketTime', 'earningsTimestamp',
                 'earningsTimestampStart', 'expiration', 'lastTradeDate']
    fs_df = fs_df.fillna(0)
    q_df = q_df.fillna(0)
    cols_to_date(fs_df, date_cols)
    cols_to_date(q_df, date_cols)

    # merge options and quotes
    merged_df = q_df.merge(fs_df, how='inner',
                           left_on=['symbol', 'regularMarketTime'],
                           right_on=['underlyingSymbol', 'lastTradeDate'])
    return merged_df


def transform_ds(df):
    # consider passing common fields as a map (divide by, divide col, etc)
    mod_df = df.copy()

    # transformation: divide by closing price (unit values to underlying price)
    divide_cols = ['lastPrice', 'strike', 'epsForward', 'epsTrailingTwelveMonths',
                   'bookValue', 'regularMarketDayHigh',
                   'regularMarketDayLow', 'regularMarketOpen', 'regularMarketPreviousClose']
    divided_by = 'regularMarketPrice'
    mod_df[divide_cols] = divide_by(mod_df, divide_cols, divided_by).round(4)
    mod_df.regularMarketVolume = divide_by(
        mod_df, ['regularMarketVolume'], 'averageDailyVolume10Day').round(4)

    # transformation: volume (liquidity) and size relative to the universe mean
    scale_cols = ['regularMarketVolume', 'averageDailyVolume10Day',
                  'averageDailyVolume3Month', 'marketCap']
    scaler_pipe = make_pipeline(StandardScaler())
    mod_df[scale_cols] = scaler_pipe.fit_transform(mod_df[scale_cols])

    mod_df.regularMarketTime = pd.to_datetime(mod_df.regularMarketTime)
    mod_df.expiration = pd.to_datetime(mod_df.expiration)
    delta = (mod_df.expiration - mod_df.regularMarketTime).dt.days
    mod_df.expiration = delta.apply(time_delta_to_years).round(4)

    return mod_df


def top_bottom_selection(df, key, sortKey, items):
    ranked_ret = df[[key,sortKey]].sort_values(by=sortKey)
    keyList = ranked_ret[key].head(items).tolist() + ranked_ret[key].tail(items).tolist()
    return df[df[key].isin(keyList)]


def filter_ds(df):
    # filter data by expiration, strike, liquidity and active contracts
    oi_mask = df.openInterest > 100
    vol_mask = df.volume > 10
    itm_mask = (df.inTheMoney == False) & (
        ((df.strike > 0.8) & (df.strike < 0.95)) |
        ((df.strike > 1.05) & (df.strike < 1.2))
    )
    exp_mask = (df.expiration > 90/365)
    return df[oi_mask & vol_mask & itm_mask & exp_mask]


def prepare_for_training(df, cols_to_keep, y_col):
    num_cols = df[cols_to_keep].select_dtypes(include=['float64'])
    order_cols = [x for x in num_cols if x != y_col]
    order_cols.extend([y_col])
    return df[order_cols]

# subset of columns we want to retain
# quote columns
to_keep = ['regularMarketTime', 'regularMarketVolume',
           'epsForward', 'epsTrailingTwelveMonths', 'regularMarketDayHigh',
           'regularMarketDayLow', 'regularMarketOpen', 'regularMarketPreviousClose',
           'averageDailyVolume10Day', 'averageDailyVolume3Month', 'marketCap',
           'bookValue', 'fiftyDayAverageChangePercent', 'fiftyTwoWeekHighChangePercent',
           'fiftyTwoWeekLowChangePercent', 'trailingAnnualDividendYield',
           'twoHundredDayAverageChangePercent']
# option columns
to_keep.extend(['underlyingSymbol', 'type',
                'expiration', 'impliedVolatility', 'lastPrice',
                'strike', 'contractSymbol'])
target_col = 'lastPrice'


# prediction and analysis functions
def update_deltas(ml_df, predictions, pred_col, var_col):
    post_pred_df = ml_df.copy()
    post_pred_df.loc[:, pred_col] = predictions
    post_pred_df.loc[:, var_col] = post_pred_df[pred_col] - \
        post_pred_df.lastPrice
    return post_pred_df


def rows_left(x, y): return print('{} left after {}'.format(len(x), y))


def rows_left_by_type(x, y, z): return print(
    '{}: {} calls, {} puts'.format(len(x), len(y), len(z)))


def run_options_recommendation():
    # Date ranges for analysis
    dates = read_dates('option')
    predict_days = -1
    predict_dates, train_dates = dates[predict_days:], dates[:predict_days]
    train_dates, predict_dates
    today_date = date.today()

    # load contracts for predict date
    pred_quote_frame = load_consol_quotes(predict_dates)
    pred_full_set = load_consol_options(predict_dates)

    # load the prediction date data
    print('{} options {} quotes'.format(
        pred_full_set.underlyingSymbol.count(),
        pred_quote_frame.symbol.count()))

    # merge, filter and transform
    pred_all_df = merge_ds(pred_full_set, pred_quote_frame)
    rows_left(pred_all_df, 'merge')
    pred_all_df = transform_ds(pred_all_df)
    rows_left(pred_all_df, 'transform')
    pred_all_df = filter_ds(pred_all_df)
    rows_left(pred_all_df, 'filter')
    pred_all_df = top_bottom_selection(pred_all_df, 'symbol', 'fiftyTwoWeekLowChangePercent', 25)
    rows_left(pred_all_df, '2nd filter')

    option_types = ['call', 'put']
    recommendations = pd.DataFrame()

    adjUpsideCol = 'adjustedUpside'
    recommendationDateCol = 'recommendationDate'

    ANALYSIS_COLS = ['contractSymbol', 'type', 'underlyingSymbol',
        'regularMarketPrice', 'impliedVolatility', 'expiration',
        'strike','lastPrice','prediction', 'predictionDelta',
        adjUpsideCol, recommendationDateCol]
    PRICING_COLS = ['lastPrice', 'prediction', 'predictionDelta']

    PREDICTION_COL = 'prediction'
    DELTA_COL = 'predictionDelta'

    for OPTION_TYPE in option_types:
        ml_df = pred_all_df[pred_all_df.type == OPTION_TYPE]
        ml_train_df = prepare_for_training(ml_df, to_keep, target_col)
        rows_left(ml_train_df, OPTION_TYPE)

        X = ml_train_df.loc[:, ml_train_df.columns[:-1]].values
        Y = ml_train_df.loc[:, target_col].values

        fname = "{}_ML_model.pkl".format(OPTION_TYPE)
        model = joblib.load(fname)
        predictions = model.predict(X)
        print('Predicted {} RMSE {}'.format(
            OPTION_TYPE, np.sqrt(mean_squared_error(Y, predictions))))

        post_pred_df = update_deltas(
            ml_df, predictions, PREDICTION_COL, DELTA_COL)
        post_pred_df[PRICING_COLS] = (
            post_pred_df[PRICING_COLS].T * \
            pred_all_df.loc[post_pred_df.index].regularMarketPrice).T
        ajdustedUpside = (post_pred_df.predictionDelta / post_pred_df.lastPrice) * \
            post_pred_df.expiration * (1 - (1 - post_pred_df.strike).abs())
        post_pred_df[adjUpsideCol] = ajdustedUpside
        post_pred_df[recommendationDateCol] = str(today_date)
        final = post_pred_df[ANALYSIS_COLS]
        recommendations = recommendations.append(final)

    recommendations = recommendations.sort_values(by=adjUpsideCol,
        ascending=False).head(12).round(3)
    path = get_path('option_recommendation')
    csv_store(
        recommendations, path,
        csv_ext.format('options-' + str(today_date)), True)
