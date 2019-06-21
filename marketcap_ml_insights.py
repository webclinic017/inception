
from utils.basic_utils import csv_load, load_config
from utils.BaseDS import BaseDS
import pandas as pd

context = load_config('./utils/marketcap_context.json')

base_ds = BaseDS(
    context['tmp_path'], 
    context['px_vol_ds'], 
    load_ds=True, 
)

dates = read_dates('quote')
tgt_date = dates[-1] # last date saved in S3
print(f'Target date: {tgt_date}')

quotes = load_csvs('quote_consol', [tgt_date])
profile = load_csvs('summary_detail', ['assetProfile'])
quotes.set_index('symbol', drop=False, inplace=True)
profile.set_index('symbol', drop=False, inplace=True)

pred_df = pd.read_csv(
    csv_load(f'{context["s3_path"]}{base_ds.tgt_date}'), 
    parse_dates=True)
pred_df.set_index(list(pred_df.columns[:2]), inplace=True)
pred_df = pred_df / 10**9

# one sample company
idx = pd.IndexSlice
symbol = 'PH'
pred_df.loc[idx[:, symbol], :].iloc[:,-2:].droplevel(1).plot(rot=45)

# last predicted date
look_back_dates = pred_df.index.levels[0][-1:]

# port_df = pd.read_csv('portfolio.csv', index_col=[0])
# port_df = pd.read_csv('inference.csv', index_col=[0])
# pos_col = 'shares'
# long_list = port_df.loc[port_df[pos_col] > 0].index
# short_list = port_df.loc[port_df[pos_col] < 0].index

# long_exp = False; cheap = True
# filtered_df = pred_df.loc[idx[look_back_dates, long_list if long_exp else short_list], :]
# premdisc_df = (filtered_df.iloc[:, 1] / filtered_df.iloc[:, 0])
# mask = premdisc_df < 1 if cheap else premdisc_df > 1
# symbol_list = premdisc_df.loc[mask].sort_values().reset_index()
# symbol_list
# for symbol in (symbol_list.symbol):
#     pred_df.loc[idx[:, symbol], :].iloc[:,-2:].droplevel(1).plot(title=symbol, rot=45)

# REVIEW ENTIRE UNVERSE
eq_val_df = pred_df.loc[idx[look_back_dates, :], :].copy()
eq_val_df = (eq_val_df.iloc[:, 1] / eq_val_df.iloc[:, 0])
p1, p3 = eq_val_df.quantile([0.1,0.9]).values
central_df = eq_val_df.loc[(eq_val_df > p1) & (eq_val_df < p3)].sort_values().droplevel(0)
val_col = 'premDisc'
central_df.name = val_col
summary = pd.concat([
    central_df,
    quotes.loc[central_df.index, ['shortName', 'marketCap']],
    profile.loc[central_df.index, ['sector', 'industry']]], axis=1)
summary = summary.loc[summary['marketCap'] > 5*10**9]
# large_ind = summary.groupby(by=['industry']).count().sort_values(by=[val_col]).tail(50)
tgt_cos = summary.loc[(summary[val_col] > 1)].sort_values(by=[val_col])
tgt_cos.groupby(by=['industry']).count().sort_values(
    by=val_col, ascending=False)
tgt_cos.loc[tgt_cos.industry.isin(['Diversified Industrials'])]