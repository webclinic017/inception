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
    tickers='All')
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
# ls_dict = {True: 0.84, False: -0.36}
ls_dict = {True: 0.5, False: -0.5}
leverage = (abs(ls_dict[True]) + abs(ls_dict[False]))
nbr_positions = 10
portfolio = 900000
amount = portfolio / nbr_positions * leverage
long = True
# stop losses dont seem to help
as_of_date = -1
# pick most frequent predictions within X study period
watch_overtime = True
study_period = -10
# cut off
min_confidence = 0.9
# percent of time in the list during study period
period_tresh = 0.5

# %% AI portfolio - one period
super_list = []
most_freq_df = None
for long in [True, False]:
    span = 3
    pred_classes = labels_list[:span] if long else labels_list[-span:]
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
    symbols = list(set(top_pos.symbol))
    last_pred_df = pred_df.loc[pred_df.index[-1],:]
    rec_summary_df = last_pred_df.loc[last_pred_df.symbol.isin(symbols), ['symbol', 'pred_class', 'pred_label', 'confidence']]
    rec_summary_df = rec_summary_df.reset_index().set_index('symbol')

    print(f'{len(symbols)} {"LONG" if long else "SHORT"} Symbols, {symbols}')

    # Share allocation
    show_cols = [
        'shortName', 'regularMarketPrice',
        'averageDailyVolume3Month', 'marketCap']
    quote_alloc = quotes.loc[symbols, show_cols]
    show_cols = ['sector', 'industry']
    profile_alloc = profile.loc[symbols, show_cols]

    w = 1 / nbr_positions * ls_dict[long]
    allocation = portfolio * w
    alloc_df = (allocation / quotes.loc[symbols, ['regularMarketPrice']]).round(0)
    alloc_df['dollarValue'] = alloc_df * quotes.loc[symbols, ['regularMarketPrice']]
    alloc_df.columns = ['shares', 'dollarValue']
    alloc_df = pd.concat([rec_summary_df, alloc_df, quote_alloc, profile_alloc], axis=1)
    super_list.append(alloc_df)

alloc_df = pd.concat(super_list, axis=0)
# alloc_df['pred_date'] = date.today()

s3_store = context['s3_store']
s3_portfolio_path = context['s3_portfolio_path']
if s3_store:
    # store in S3
    s3_df = alloc_df.reset_index(drop=False)
    # rename_col(s3_df, 'index', 'pred_date')
    csv_store(s3_df, s3_portfolio_path, csv_ext.format(f'{tech_ds.tgt_date}'))
else:
    alloc_df.to_csv(f'{str(date.today())}.csv')


# # %% historical index for predictions
# get_ind_index(clean_px[symbols], tail=252, name='^PORT')['^PORT'].plot(
#     title='Historical Performance of Portfolio'
# );

# %% what's the average class and confidence level by sector?
# pred_symbols_df = pred_df.reset_index().set_index('symbol')
# pred_desc = profile.loc[pred_symbols_df.index, ['sector', 'industry']]
# show_cols = ['pred_date', 'pred_class', 'pred_label', 'confidence', 'sector', 'industry']
# pred_clean = pd.concat([pred_symbols_df, pred_desc], axis=1)[show_cols]
# idx = pd.IndexSlice

# # average class and confidence by sector
# symbol_date_pred_df = pred_clean.reset_index().set_index(['symbol', 'pred_date'])
# last_date = symbol_date_pred_df.index.levels[1].unique()[-1]
# last_pred = symbol_date_pred_df.loc[idx[:, last_date], :]
# # sector level count
# last_pred.groupby(by=['sector']).count().sort_values(by=['pred_class'], ascending=False).dropna()
# # sector level median
# last_pred.groupby(by=['sector']).median().sort_values(by=['pred_class'], ascending=False).dropna()
# # sector and industry deeper dive
# last_pred.groupby(by=['sector', 'industry']).median().sort_values(by=['sector'], ascending=False).dropna()
# # sector and industry deeper dive
# sel_sector = 'Technology'
# sel_industry = 'Software - Infrastructure'
# mask = (last_pred.sector == sel_sector) & (last_pred.industry == sel_industry)
# last_pred.loc[mask, :]

# # historical charts of average class and confidence by sector
# groups = pred_clean.groupby(by=['sector', 'pred_date']).median()
# for i in groups.index.levels[0]:
#     groups.loc[idx[i, :], :].plot(title=i, secondary_y='confidence')

# %% By sector
# by_sect = alloc_df.groupby(by=['sector']).sum().loc[:,'dollarValue'].sort_values()
# (by_sect / amount).plot.bar()

# # %% By industry
# by_ind = alloc_df.groupby(by=['industry']).sum().loc[:,'dollarValue'].sort_values()
# (by_ind / amount).plot.bar()

# # %%
# finstats.loc[symbols]
# list(quotes.columns)

# %% check one company confidence levels
# curr_port = ['CRM']
# pred_df.loc[pred_df.symbol.isin(curr_port), :]

# # %% check one company confidence levels
# most_freq_df
# pred_df.loc[pred_df.symbol.isin(['SE']), 'confidence'].tail(10).mean()

# # %% double check mean confidence over study period
# df = pred_df.loc[pred_df.symbol.isin(symbols), ['symbol', 'pred_class', 'confidence']]
# df = df.set_index('symbol', append=True).unstack().tail(10)
# df['confidence'].plot(legend=False)
# df['confidence'].mean().sort_values()
# symbols

# %%
