from basic_utils import *
from pricing import *

def co_price_mom_ds(symbol):
    # Retrieves historical pricing
    secpx = get_symbol_pricing(symbol, freq)
    secpx.set_index(secpx.index.astype(np.datetime64), inplace=True)
    closepx = secpx['close']

    # industry, sector and market performance
    row = profile[profile.symbol == symbol]
    sec_sector, sec_industry = row.iloc[0].sector, row.iloc[0].industry

    # Note: since the universe is limited industry and sector
    # are not a good representation, need to expand universe
    # for better learning
    sec_index = to_index_form(closepx, symbol)
    symbols = profile[profile.industry == sec_industry].symbol.tolist()
    industry_index = get_ind_index(px_set[symbols], '1d', tail, ind_idx_ticker)[[ind_idx_ticker]]
    symbols = profile[profile.sector == sec_sector].symbol.tolist()
    sector_index = get_ind_index(px_set[symbols], '1d', tail, sect_idx_ticker)[[sect_idx_ticker]]
    market_index = to_index_form(get_symbol_pricing(market_etf, freq, 'close').tail(tail), market_etf)

    index_df = pd.DataFrame()
    index_df = index_df.append(sec_index)
    index_df[ind_idx_ticker] = industry_index
    index_df[sect_idx_ticker] = sector_index
    index_df[market_etf] = market_index

    ind_sect_spy = index_df[[ind_idx_ticker, sect_idx_ticker, market_etf]]
    index_df[['indChg1m', 'sectChg1m', 'spyChg1m']] = ind_sect_spy.pct_change(periods=20)
    index_df[['indChg3m', 'sectChg3m', 'spyChg3m']] = ind_sect_spy.pct_change(periods=60)
    index_df[['indChg6m', 'sectChg6m', 'spyChg6m']] = ind_sect_spy.pct_change(periods=180)

    # apply 20sma upper and lower std bands, # stds from file
    secpx = apply_std_boundaries(secpx, 'close', 20, stds)
    secpx['pxPercStdUB'] = closepx / secpx['sma20ub']
    secpx['pxPercStdLB'] = closepx / secpx['sma20lb']

    # Volume averages
    volume = secpx['volume']
    secpx['volMa10'] = volume.rolling(20).mean()
    secpx['volMa60'] = volume.rolling(60).mean()

    # Volume as a % of 10 and 60 day average
    secpx['volPercMa10'] = volume / secpx['volMa10']
    secpx['volPercMa60'] = volume / secpx['volMa60']

    # Price momentum transformations
    secpx['pxMa20'] = closepx.rolling(20).mean()
    secpx['pxMa50'] = closepx.rolling(50).mean()
    secpx['pxMa200'] = closepx.rolling(200).mean()

    # closing pricing as % of 20, 50 and 200 day average
    secpx['pxPercMa20'] = closepx / secpx['pxMa20']
    secpx['pxPercMa50'] = closepx / secpx['pxMa50']
    secpx['pxPercMa200'] = closepx / secpx['pxMa200']

    # historical returns for 1, 3, and 6 months
    secpx['chg1m'] = closepx.pct_change(periods=20)
    secpx['chg3m'] = closepx.pct_change(periods=60)
    secpx['chg6m'] = closepx.pct_change(periods=180)

    # Forward returns, 1w, 1m, 3m
    secpx['fwdChg1w'] = closepx.pct_change(periods=-5)
    secpx['fwdChg1m'] = closepx.pct_change(periods=-20)
    secpx['fwdChg3m'] = closepx.pct_change(periods=-60)

    # Relative strength to industry, sector and market
    secpx['rs1mInd'] = (secpx['chg1m'] / index_df['indChg1m'])
    secpx['rs3mInd'] = (secpx['chg3m'] / index_df['indChg3m'])
    secpx['rs6mInd'] = (secpx['chg6m'] / index_df['indChg6m'])

    secpx['rs1mSect'] = (secpx['chg1m'] / index_df['sectChg1m'])
    secpx['rs3mSect'] = (secpx['chg3m'] / index_df['sectChg3m'])
    secpx['rs6mSect'] = (secpx['chg6m'] / index_df['sectChg6m'])

    secpx['rs1mSPY'] = (secpx['chg1m'] / index_df['spyChg1m'])
    secpx['rs3mSPY'] = (secpx['chg3m'] / index_df['spyChg3m'])
    secpx['rs6mSPY'] = (secpx['chg6m'] / index_df['spyChg6m'])

    # seasonality analysis
    ss_df = closepx.pct_change().resample('M').sum().to_frame()
    ss_df['year'], ss_df['month'] = ss_df.index.year, ss_df.index.month
    ss_df = ss_df.pivot_table(index='year', columns='month').mean()
    ss_pos = [(x, (x+1) if not (x+1) // 12 else 0,
         x+2 if not (x+2) // 12 else x - 10) for x in range(12)]

    # apply seasonality, mean return of curr month plus next two
    secpx['month'] = secpx.index.month
    secpx['fwdSSRet'] = secpx.loc[:].month.apply(
    fwd_ss_ret, args=(ss_df['close'], ss_pos,))

    # normalized columns for ML training, still has outliers
    ml_ds_cols = secpx.describe().loc['50%'][secpx.describe().loc['50%'] < 5].index.tolist()

    return secpx[ml_ds_cols]

# TAKES LONG: all pricing loaded, should do only once
symbols_list = config['benchmarks'] + config['sectors'] + config['companies']
px_set = get_mults_pricing(symbols_list, freq, 'close')

# all equities
# excl_list = ['CELG']
# profile.drop(profile[profile.symbol.isin(excl_list)].index, inplace=True)
all_equities = quotes[quotes.quoteType == 'EQUITY'].symbol.unique()
eqty_symbols = profile[profile.symbol.isin(all_equities)].symbol.unique().tolist()

treshold = (8 * 252) # at least 8 years of pricing
tail = (10 ** 4) # no tail
count_df = px_set[eqty_symbols].describe().loc['count']
ds_symbols = count_df[count_df > treshold].index.tolist()

ds_name = 'co_price_mom_ds'
update_fmt = 'Added {} to {} dataset'

# Running for entire universe takes a while
ml_ds_df = pd.DataFrame()
for s in ds_symbols:
    print(update_fmt.format(s, ds_name))
    ml_ds = co_price_mom_ds(s)
    ml_ds_df = ml_ds_df.append(ml_ds.copy(), sort=False)

med_cutoff = ml_ds_df.describe().loc['50%']
ml_ds_cols = med_cutoff[med_cutoff < 3].index.tolist()
ml_ds_df = ml_ds_df[ml_ds_cols]

# Discretize forward returns into consistent classes
class_cols = ['fwdChg1w', 'fwdChg1m', 'fwdChg3m']
cut_range = [-1, -0.05, .0, .02, .09, 1.]
fwd_ret_labels = ["bear", "short", "neutral", "long", "bull"]
for c in class_cols: ml_ds_df[c] = pd.cut(ml_ds_df[c], cut_range, labels=fwd_ret_labels)

# Drop records showing nans
ml_ds_df.dropna(inplace=True)

print(update_fmt.format(len(ml_ds_df), ds_name))
csv_store(ml_ds_df, 'training/', csv_ext.format('co_price_mom_ds'))
