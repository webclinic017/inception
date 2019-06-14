# %%
import os
import pandas as pd
from datetime import date
import matplotlib as mpl
from matplotlib import pyplot as plt

from utils.basic_utils import csv_store, csv_load, csv_ext, numeric_cols
from utils.pricing import dummy_col, px_fwd_ret, get_ind_index, discret_rets
from utils.pricing import rename_col
from utils.fundamental import chain_outlier
from utils.TechnicalDS import TechnicalDS
from utils.pred_utils import get_top_predictions, get_most_frequent_preds, get_study_date_range, stop_loss

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

pd.options.display.float_format = '{:,.2f}'.format
mpl.rcParams['font.size'] = 8
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'large'

# %% 
context = {
    'ml_path': './ML/',
    'model_name': 'micro_TF.h5',
    'tmp_path': './tmp/',
    'px_vol_ds': 'universe-px-vol-ds.h5',
    'look_ahead': 120,
    'look_back': 252*15,
    'load_ds': True,
    'verbose': True,
    's3_store': True,
    's3_pred_path': 'recommend/micro_ML/',
    's3_portfolio_path': 'recommend/equities/',

}

# %%
tech_ds = TechnicalDS(
    context['tmp_path'],
    context['px_vol_ds'],
    load_ds=context['load_ds'],
    look_ahead=context['look_ahead'],
    max_draw_on=True,
    tickers='All',
)

tgt_date = tech_ds.tgt_date
quotes = tech_ds.quotes
profile = tech_ds.profile
keystats = tech_ds.keystats
finstats = tech_ds.finstats
clean_px, labels = tech_ds.clean_px, tech_ds.forward_return_labels
labels_list = list(reversed(range(len(tech_ds.forward_return_labels))))

# Read today's predictions from S3
s3_path = context['s3_pred_path']
pred_df = pd.read_csv(
    csv_load(f'{s3_path}{tgt_date}'),
    index_col='pred_date',
    parse_dates=True)
pred_df.info()
print('Prediction distribution')
print(pd.value_counts(pred_df.pred_label) / pd.value_counts(pred_df.pred_label).sum())
pred_df.tail()

# %% Assumptions
# enable long or short
ls_dict = {True: 0.84, False: -0.36}
leverage = (abs(ls_dict[True]) + abs(ls_dict[False]))
amount = 875000/6 * leverage
long = True
# stop losses dont seem to help
loss_protection = False
max_loss = 0.1
# how many days to hold / rebalance
holding_period = 120
as_of_date = -1
# pick most frequent predictions within X study period
watch_overtime = True
study_period = -10
# cut off
min_confidence = 0.95
# percent of time in the list during study period
period_tresh = 0.5
nbr_positions = 10
look_ahead, look_back = context['look_ahead'], context['look_back']

# %% AI portfolio - one period
super_list = []
most_freq_df = None
for long in [True, False]:
    pred_classes = labels_list[:2] if long else labels_list[-2:]
    top_pred = get_top_predictions(pred_df, as_of_date, pred_classes, min_confidence)
    study_dates = get_study_date_range(pred_df, as_of_date, study_period)
    most_freq_df = get_most_frequent_preds(
        pred_df, study_dates, top_pred,
        pred_classes, period_tresh)
    if watch_overtime:
        top_pos = most_freq_df
    else:
        top_pos = top_pred.loc[
            top_pred.pred_class.isin(pred_classes) &
            top_pred.confidence > min_confidence]
    symbols = list(top_pos.symbol)
    print(f'{len(symbols)} {"LONG" if long else "SHORT"} Symbols, {symbols}')

    # Share allocation
    show_cols = [
        'shortName', 'regularMarketPrice',
        'averageDailyVolume3Month', 'marketCap']
    quote_alloc = quotes.loc[symbols, show_cols]
    show_cols = ['sector', 'industry']
    profile_alloc = profile.loc[symbols, show_cols]

    w = 1 / nbr_positions * ls_dict[long] / leverage
    allocation = amount * w
    alloc_df = (allocation / quotes.loc[
        symbols, ['regularMarketPrice']]).round(0)
    alloc_df['dollarValue'] = alloc_df * quotes.loc[
        symbols, ['regularMarketPrice']]
    alloc_df.columns = ['shares', 'dollarValue']
    alloc_df = pd.concat([alloc_df, quote_alloc, profile_alloc], axis=1)
    super_list.append(alloc_df)

alloc_df = pd.concat(super_list, axis=0)
alloc_df['pred_date'] = date.today()

s3_store = context['s3_store']
s3_portfolio_path = context['s3_portfolio_path']

if s3_store:
    # store in S3
    s3_df = alloc_df.reset_index(drop=False)
    # rename_col(s3_df, 'index', 'pred_date')
    csv_store(s3_df, s3_portfolio_path, csv_ext.format(f'{tech_ds.tgt_date}'))
else:
    alloc_df.to_csv(f'{str(date.today())}.csv')


# %% historical index for predictions
# get_ind_index(clean_px[symbols], tail=252, name='^PORT')['^PORT'].plot(
#     title='Historical Performance of Portfolio'
# );

# %% By sector
# by_sect = alloc_df.groupby(by=['sector']).sum().loc[:,'dollarValue'].sort_values()
# (by_sect / amount).plot.bar()

# %% By industry
# by_ind = alloc_df.groupby(by=['industry']).sum().loc[:,'dollarValue'].sort_values()
# (by_ind / amount).plot.bar()

# %%
# finstats.loc[symbols]
# list(quotes.columns)

# %% check one company confidence levels
# most_freq_df
# pred_df.loc[pred_df.symbol.isin(['CHTR']), 'confidence'].tail(10).mean()

# %% double check mean confidence over study period
# df = pred_df.loc[pred_df.symbol.isin(symbols), ['symbol', 'pred_class', 'confidence']]
# df = df.set_index('symbol', append=True).unstack().tail(10)
# df['confidence'].plot(legend=False)
# df['confidence'].mean().sort_values()
# symbols