# imports
from utils.basic_utils import *
import numpy as np

# lambdas
freq_dist = lambda df, col, tail: df[col].tail(tail).value_counts(bins=12, normalize=True).sort_index()
shorten_name = lambda x: "".join([str.upper(z[:3]) for z in x])
roll_vol = lambda df, rw: (df.rolling(rw).std() * pow(252, 1/2))
fwd_ss_ret = lambda x, df, arr: df.loc[[y for y in arr[x-1] if y in df.index.tolist()]].mean()
sign_compare = lambda x, y: abs(x) // y if x > y else -(abs(x) // y) if x < -y else 0
pos_neg = lambda x: -1 if x < 0 else 1

# helper methods
def build_px_struct(data_dict, freq):
    dt = date if freq == '1d' else datetime
    dates = [dt.fromtimestamp(x) for x in data_dict['timestamp']]
    hist_pricing = data_dict['indicators']['quote'][0]
    H = hist_pricing['high']
    L = hist_pricing['low']
    O = hist_pricing['open']
    C = hist_pricing['close']
    V = hist_pricing['volume']
    price_dict = {'high': H,'low': L,'open': O,'close' : C, 'volume': V}
    df = pd.DataFrame(price_dict, index=dates)
    return df

def get_rt_pricing(symbol, freq='1d', prange='10d', cols=None):
    data_dict = get_pricing(symbol, freq, prange)
    df = build_px_struct(data_dict, freq)
    cols = df.columns if cols is None else cols
    return df[cols].dropna()

def get_symbol_pricing(symbol, freq='1d', cols=None):
    path = config['pricing_path'].format(freq)
    data_dict = json_load(path + json_ext.format(symbol))
    df = build_px_struct(data_dict, freq)
    cols = df.columns if cols is None else cols
    return df[cols].dropna()

def get_mults_pricing(symbols, freq, col='close'):
    group_pricing = pd.DataFrame()
    for n, t in enumerate(symbols):
        try:
            sec_hp = get_symbol_pricing(t, freq, [col])
            sec_hp.rename(columns={col: t}, inplace=True)
            if n == 0: group_pricing = pd.DataFrame(sec_hp)
            else: group_pricing[t] = sec_hp
            print("Retrieved pricing for {0}".format(t))
        except Exception as e:
            print("Exception on {0}\n{1}".format(t, e))
    return group_pricing

def apply_std_boundaries(df, col='close', window=30, stds=2):
    sma = df[col].rolling(window).mean()
    smastd = df[col].rolling(window).std()
    smaub = sma + smastd*stds
    smalb = sma - smastd*stds
    df['sma' + str(window)] = sma
    df['sma' + str(window) + 'ub'] = smaub
    df['sma' + str(window) + 'lb'] = smalb
    df['sma' + str(window) + 'bw'] = smastd*stds / sma
    return df

def get_ind_index(closepx, freq='1d', tail=60, name='^IND'):
    closepx = closepx.tail(tail)
    memb_count = len(closepx.columns.tolist())
    eq_wgt = 1 / memb_count
    closepx.set_index(pd.DatetimeIndex(closepx.index), inplace=True)
    comp_indexed = (closepx.pct_change() + 1).cumprod()
    comp_indexed.iloc[0] = 1
    # comp_indexed.pct_change(), closepx.pct_change() # CHECK, should be the same
    comp_indexed[name] = (comp_indexed * eq_wgt).sum(axis=1)
    return comp_indexed

def to_index_form(df, name):
    dt_index = pd.DatetimeIndex(df.index)
    idx = pd.DataFrame((df.pct_change() + 1).cumprod().values,
        index=dt_index, columns=[name])
    idx.iloc[0] = 1
    return idx

def beta(df):
    # first column is the market
    X = df.values[:, [0]]
    # prepend a column of ones for the intercept
    X = np.concatenate([np.ones_like(X), X], axis=1)
    # matrix algebra
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:])
    return pd.Series(b[1], df.columns[1:], name='Beta')

def get_statspc_dates(df, treshold):
    mask = df.abs() / df.std() > treshold
    return df[mask][(df[mask].sum(axis=1) != 0).values]

def eq_wgt_attribution(comp_idx, index_col, resample_period):
    rp_delta = comp_idx.sub(comp_idx[index_col], axis='rows').iloc[:,:-1]
    resampled = (rp_delta/len(rp_delta.columns)).resample(resample_period, level=0).sum()
    return resampled[resampled.iloc[-1].sort_values(ascending=False).index]

def rank_roll_perf(df, roll_window):
    ind_ranked = df.round(2).rank(axis=1)
    rolled_ranked = ind_ranked.rolling(roll_window).mean()
    show = rolled_ranked.iloc[-1].sort_values().index.tolist()
    return rolled_ranked[show]

def get_left_right(alist, sl):
    # mods large list into left / right (top and bottom quartiles)
    be = len(alist) // sl
    left, right = be + 1 if be * sl < len(alist) else be, be
    return left, right

def cutoff_tresh_cols(df, col, th):
    med_co_df = df.describe().loc[col]
    return med_co_df[med_co_df < th].index.tolist()

def co_price_mom_ds(symbol, px_set):
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
    secpx['pxPercStdUB'] = closepx / secpx['sma20ub'] - 1
    secpx['pxPercStdLB'] = closepx / secpx['sma20lb'] - 1

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
    secpx['pxPercMa20'] = closepx / secpx['pxMa20'] - 1
    secpx['pxPercMa50'] = closepx / secpx['pxMa50'] - 1
    secpx['pxPercMa200'] = closepx / secpx['pxMa200'] - 1

    # historical returns for 1, 3, and 6 months
    secpx['chg1m'] = closepx.pct_change(periods=20)
    secpx['chg3m'] = closepx.pct_change(periods=60)
    secpx['chg6m'] = closepx.pct_change(periods=180)

    # Forward returns, 1w, 1m, 3m
    secpx['fwdChg1w'] = closepx.pct_change(5).shift(-5)
    secpx['fwdChg1m'] = closepx.pct_change(20).shift(-20)
    secpx['fwdChg3m'] = closepx.pct_change(60).shift(-60)

    # Relative strength to industry, sector and market
    secpx['rs1mInd'] = (secpx['chg1m'] - index_df['indChg1m'])
    secpx['rs3mInd'] = (secpx['chg3m'] - index_df['indChg3m'])
    secpx['rs6mInd'] = (secpx['chg6m'] - index_df['indChg6m'])

    secpx['rs1mSect'] = (secpx['chg1m'] - index_df['sectChg1m'])
    secpx['rs3mSect'] = (secpx['chg3m'] - index_df['sectChg3m'])
    secpx['rs6mSect'] = (secpx['chg6m'] - index_df['sectChg6m'])

    secpx['rs1mSPY'] = (secpx['chg1m'] - index_df['spyChg1m'])
    secpx['rs3mSPY'] = (secpx['chg3m'] - index_df['spyChg3m'])
    secpx['rs6mSPY'] = (secpx['chg6m'] - index_df['spyChg6m'])

    # seasonality analysis
    ss_df, ss_pos = get_pct_chg_seasonality(closepx, 'M')

    # apply seasonality, mean return of curr month plus next two
    secpx['month'] = secpx.index.month
    secpx['fwdSSRet'] = secpx.loc[:].month.apply(fwd_ss_ret, args=(ss_df, ss_pos,))
    secpx.drop(columns=['month'], inplace=True)

    # normalized columns for ML training, still has outliers
    # ml_ds_cols = secpx.describe().loc['50%'][secpx.describe().loc['50%'] < 5].index.tolist()

    return secpx

def get_pct_chg_seasonality(df, rule):
    ss_df = df.pct_change().resample(rule).sum().to_frame()
    ss_df['year'], ss_df['month'] = ss_df.index.year, ss_df.index.month
    ss_df = ss_df.pivot_table(index='year', columns='month').mean()
    ss_pos = [(x, (x+1) if not (x+1) // 12 else 0,
         x+2 if not (x+2) // 12 else x - 10) for x in range(12)]
    return ss_df.loc[('close'),:], ss_pos

# contextual variables, can be configured externally
market_etf = '^GSPC'
freq, tail = '1d', 10**5
window, stds = 20, 1.75
dates = read_dates('quote')
tgt_date = [dates[-1]] # last date saved in S3
sl = 3 # to slice large lists in top / bottom chunks
show = ['symbol','sector', 'industry']
sect_idx_ticker, ind_idx_ticker = '^SECT', '^IND'

# latest_quotes = load_csvs('quote_consol', tgt_date)
quotes = load_csvs('quote_consol', tgt_date)
profile = load_csvs('summary_detail', ['assetProfile'])

industries = profile[show].sort_values(by='industry').industry.dropna().unique().tolist()
left, right = get_left_right(industries, sl)
