# imports
from basic_utils import *
import numpy as np

# lambdas
freq_dist = lambda df, col, tail: df[col].tail(tail).value_counts(bins=12, normalize=True).sort_index()
shorten_name = lambda x: "".join([str.upper(z[:3]) for z in x])
roll_vol = lambda df, rw: (df.rolling(rw).std() * pow(252, 1/2))
fwd_ss_ret = lambda x, df, arr: df.loc[[y for y in arr[x-1] if y in df.index.tolist()]].mean()

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
        print("Getting pricing for {0}".format(t))
        sec_hp = get_symbol_pricing(t, freq,[col])
        sec_hp.rename(columns={col: t}, inplace=True)
        if n == 0: group_pricing = pd.DataFrame(sec_hp)
        else: group_pricing[t] = sec_hp
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

# contextual variables, can be configured externally
market_etf = 'SPY'
freq = '1d'
window, stds = 20, 1.75
dates = read_dates('quote')
tgt_date = [dates[-1]] # last date saved in S3
sl = 3 # to slice large lists in top / bottom chunks

# latest_quotes = load_csvs('quote_consol', tgt_date)
quotes = load_csvs('quote_consol', tgt_date)
profile = load_csvs('summary_detail', ['assetProfile'])
show = ['symbol','sector', 'industry']
industries = profile[show].sort_values(by='industry').industry.dropna().unique().tolist()
left, right = get_left_right(industries, sl)
sect_idx_ticker, ind_idx_ticker = '^SECT', '^IND'
