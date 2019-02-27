from basic_utils import *
from pricing import *

# TAKES LONG: all pricing loaded, should do only once
excl_list = ['ORBK'] # removes tickers deleted from config.json
symbols_list = config['benchmarks'] + config['companies']
[symbols_list.remove(x) for x in excl_list]
px_set = get_mults_pricing(symbols_list, freq, 'close')

# all equities
profile.drop(profile[profile.symbol.isin(excl_list)].index, inplace=True)
all_equities = quotes[quotes.quoteType == 'EQUITY'].symbol.unique()
eqty_symbols = profile[profile.symbol.isin(all_equities)].symbol.unique().tolist()

treshold = (8 * 252) # at least 8 years of pricing
tail = (10 ** 4) # no tail
count_df = px_set[eqty_symbols].describe().loc['count']
ds_symbols = count_df[count_df > treshold].index.tolist()

ds_name = 'ds_co_price_mom_'
update_fmt = 'Added {} to {} dataset'

# Running for entire universe takes a while
ml_ds_df = pd.DataFrame()
for s in ds_symbols:
    print(update_fmt.format(s, ds_name))
    ml_ds = co_price_mom_ds(s, px_set)
    ml_ds_df = ml_ds_df.append(ml_ds.copy(), sort=False)

ml_ds_cols = cutoff_tresh_cols(ml_ds_df, '50%', 3)
ml_ds_df = ml_ds_df[ml_ds_cols]

# Discretize forward returns into consistent classes
class_cols = ['fwdChg1w', 'fwdChg1m', 'fwdChg3m']
cut_range = [-1, -0.05, .0, .02, .09, 1.]
fwd_ret_labels = ["bear", "short", "neutral", "long", "bull"]
for c in class_cols: ml_ds_df[c] = pd.cut(ml_ds_df[c], cut_range, labels=fwd_ret_labels)

Xs = [x for x in ml_ds_cols if x not in class_cols]
for y in class_cols:
    ds_cols = Xs + [y] # join Xs and Y
    fname = ds_name + y # dataset name
    ml_ds_df_y = ml_ds_df[ds_cols].dropna().sample(frac=1)
    csv_store(ml_ds_df_y, 'training/', csv_ext.format(fname))
    print(update_fmt.format(len(ml_ds_df_y), fname))
