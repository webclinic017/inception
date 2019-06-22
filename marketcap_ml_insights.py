
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

# load marketcap predictions
pred_df = pd.read_csv(
    csv_load(f'{context["s3_path"]}{base_ds.tgt_date}'), 
    parse_dates=True)
pred_df.set_index(list(pred_df.columns[:2]), inplace=True)
pred_df = pred_df / 10**9

# load technical predictions
technical_df = pd.read_csv(
    csv_load(f'recommend/micro_ML/{base_ds.tgt_date}'), 
    parse_dates=True)
technical_df.set_index(list(technical_df.columns[:2]), inplace=True)
technical_df.info()

# last predicted date
look_back_dates = pred_df.index.levels[0][-1:]

port_df = pd.read_csv('portfolio.csv', index_col=[0])
# port_df = pd.read_csv('inference.csv', index_col=[0])
pos_col = 'Position'
long_list = port_df.loc[port_df[pos_col] > 0].index
short_list = port_df.loc[port_df[pos_col] < 0].index

long_exp = True; cheap = True
filtered_df = pred_df.loc[idx[look_back_dates, long_list if long_exp else short_list], :]
premdisc_df = (filtered_df.iloc[:, 1] / filtered_df.iloc[:, 0])
mask = premdisc_df < 1 if cheap else premdisc_df > 1
symbol_list = premdisc_df.loc[mask].sort_values().reset_index()
symbol_list
# for symbol in (symbol_list.symbol):
#     pred_df.loc[idx[:, symbol], :].iloc[:,-2:].droplevel(1).plot(title=symbol, rot=45)

# REVIEW ENTIRE UNVERSE
eq_val_df = pred_df.loc[idx[look_back_dates, :], :].copy()
eq_val_df = (eq_val_df.iloc[:, 1] / eq_val_df.iloc[:, 0])
p1, p3 = eq_val_df.quantile([0.1, 0.9]).values
central_df = eq_val_df.loc[(eq_val_df > p1) & (eq_val_df < p3)].sort_values().droplevel(0)
val_col = 'premDisc'
central_df.name = val_col
summary = pd.concat([
    central_df,
    quotes.loc[central_df.index, ['shortName', 'marketCap']],
    profile.loc[central_df.index, ['sector', 'industry']]], axis=1)
summary = summary.loc[summary['marketCap'] > 5*10**9].sort_values(by=val_col)

want_long = False
tgt_cos = summary.loc[(summary[val_col] < 0.9) if want_long else (summary[val_col] > 1.1)].sort_values(by=[val_col])
tgt_cos.sort_values(by=['sector', 'industry', 'premDisc'])
# tgt_cos.groupby(by=['industry']).count().sort_values(by=val_col, ascending=False)
# tgt_cos.loc[tgt_cos.industry.isin(['Diversified Industrials'])]

# both lists, WIP
tech_df = technical_df.loc[idx[technical_df.index.levels[0][-1:], tgt_cos.index], :]
tgt_classes = [5, 4] if want_long else [0, 1]
min_confidence = 0.8
both_preds_list = tech_df.loc[tech_df.pred_class.isin(tgt_classes) & (tech_df.confidence > min_confidence)].droplevel(0).index

summary.loc[both_preds_list]

# one sample company
def plot_pred_mktcap(symbol, pred_df):
    idx = pd.IndexSlice
    plot_df = pred_df.loc[idx[:, symbol], :].iloc[:,-2:].droplevel(1)
    mean, std = plot_df['current'].describe()[['mean', 'std']]
    plot_df.plot(title=symbol, rot=45, ylim=((mean - std*3), (mean + std*3)))

symbol = 'AEP'
plot_pred_mktcap(symbol, pred_df)

for s in tgt_cos.loc[tgt_cos.sector.isin(['Utilities'])].index:
    plot_pred_mktcap(s, pred_df)