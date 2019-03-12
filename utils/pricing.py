# imports
from utils.basic_utils import *
from utils.structured import *
import numpy as np

# lambdas
freq_dist = lambda df, col, tail: df[col].tail(tail).value_counts(bins=12, normalize=True).sort_index()
shorten_name = lambda x: "^"+"_".join([str.upper(z[:4]) for z in x.replace('& ','').replace('- ','').split(' ')])
roll_vol = lambda df, rw: (df.rolling(rw).std() * pow(252, 1/2))
fwd_ss_ret = lambda x, df, arr: df.loc[[y for y in arr[x-1] if y in df.index.tolist()]].mean()
sign_compare = lambda x, y: abs(x) // y if x > y else -(abs(x) // y) if x < -y else 0
pos_neg = lambda x: -1 if x < 0 else 1
def rename_col(df, col, name): return df.rename({col: name}, axis=1, inplace=True)

# Distribution of historical exposure
def sample_wgts(y_col, sort): return (pd.value_counts(y_col) / pd.value_counts(y_col).sum())[sort]

# helper methods
def get_pricing(symbol, interval='1d', prange='5y', persist=True):
    # save pricing for a given interval and range
    dataset = 'pricing'
    point = {'s':symbol,'i':interval, 'r': prange}
    print('Getting pricing interval of {s} interval: {i}, range: {r}'.format(**point))
    # first expiration no date
    data = get_data_params('pricing', point)
    json_dict = json.loads(data)
    pricing_data = json_dict['chart']['result'][0]
    if persist:
        data = json.dumps(pricing_data)
        path = get_path(dataset, interval)
        store_s3(data, path + json_ext.format(symbol))
    return pricing_data

def build_px_struct(data_dict, freq):
    dt = date if freq == '1d' else datetime
    tz = data_dict['meta']['exchangeTimezoneName']
    dates = pd.to_datetime(
        data_dict['timestamp'], unit='s', infer_datetime_format=True)
    # dates = dates.astype(f'datetime64[ns, {tz}]')
    dates = dates.tz_localize('America/New_York')
    # dates = dates.tz_convert('America/New_York')
    hist_pricing = data_dict['indicators']['quote'][0]
    H = hist_pricing['high']
    L = hist_pricing['low']
    O = hist_pricing['open']
    C = hist_pricing['close']
    # adjC = data_dict['indicators']['adjclose'][0] if 'adjclose' in  data_dict['indicators'] else 0
    V = hist_pricing['volume']
    price_dict = {
        'high': H, 'low': L,
        'open': O, 'close': C,
        'volume': V}
    return pd.DataFrame(
        price_dict, index=dates.floor('d' if freq == '1d' else 'T'))

def get_symbol_pricing(symbol, freq='1d', cols=None):
    path = config['pricing_path'].format(freq)
    data_dict = json_load(path + json_ext.format(symbol))
    df = build_px_struct(data_dict, freq)
    cols = df.columns if cols is None else cols
    return df[cols].dropna()

def get_mults_pricing(symbols, freq='1d', col=['close']):
    super_list = []
    for n, t in enumerate(symbols):
        try:
            df = get_symbol_pricing(t, freq, col)
            rename_col(df, 'close', t)
            print("Retrieving pricing: {0}".format(t))
            # if n == 0:
            #     group_pricing = pd.DataFrame(df)
            #     continue
            super_list.append(df)
            # group_pricing = pd.concat([group_pricing, df], axis=1)
        except Exception as e:
            print("Exception, get_mults_pricing: {0}\n{1}".format(t, e))
    return pd.concat(super_list, axis=1)

def get_rt_pricing(symbol, freq='1d', prange='10d', cols=None):
    data_dict = get_pricing(symbol, freq, prange, False)
    df = build_px_struct(data_dict, freq)
    cols = df.columns if cols is None else cols
    return df[cols]

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

# Utility functions
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)

def log_as_dict(x):
    obj = x
    if not isinstance(x, (pd.Series, pd.DataFrame)):
        obj = pd.DataFrame(x)
    return obj.describe(include='all').round(3).to_dict()

# Rates transformations
def rate_feats(df, rolls=[60]):
    ndf = pd.DataFrame()

    # bps daily change
    # bps_chg = (df - df.shift(1))
    # ndf[[x + 'BpsChg' for x in df.columns]] = bps_chg
    # bps rolling change
    # for r in rolls:
    #      cum_bps_chg = (df - df.shift(1)).rolling(r).sum()
    #      ndf[[x + 'BpsChg' + str(r) for x in df.columns]] = cum_bps_chg
    # term structure spreads
    # ts_prem = (df - df.shift(1, axis=1))
    # ndf[[x + 'TSPrem' for x in df.columns[1:]]] = ts_prem.iloc[:, 1:]

    # ST (3m) vs. LT (10yr) spread
    ndf['slRateSpread'] = (df['^TNX'] - df['^IRX'])
    return ndf

def rf_feat_importance(m, df):
    return pd.DataFrame(
        {'cols':df.columns, 'imp':m.feature_importances_}
    ).sort_values('imp', ascending=False)

def show_fi(m, X, max_feats):
    importances = m.feature_importances_
    std = np.std([tree.feature_importances_ for tree in m.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for x, f in enumerate(indices):
        print("{} feature {} ({})".format(f, X.columns[indices[x]], importances[f]))
        if x >= max_feats: break

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center", )
    plt.xticks(range(len(indices)), X.columns[indices], rotation='vertical')
    plt.xlim([-1, max_feats])
    plt.show()

def show_nas(df): return df.loc[:,df.isna().sum()>0]

# discretize forward returns into classes
def discret_rets(df, treshs, classes):
    if isinstance(df, pd.Series):
        return pd.cut(df.dropna(), treshs, labels=classes)
    else:
        df.dropna(inplace=True)
        for c in df.columns: df[c] = pd.cut(df[c], treshs, labels=classes)
    return df

# Generic price momentum transformations
def px_mom_feats(df, s, stds=1, invert=False, incl_px=False, rolls=[20,60,120], incl_name=True):
    ndf = pd.DataFrame()
    if invert: df = 1 / df
    c,o,l,h = df['close'], df['open'], df['low'], df['high']
    c1ds, pctChg = c.shift(1), c.pct_change()
    if incl_px: ndf[s + 'Close'] = c
    ticker = s if incl_name else ''
    ndf[ticker+'PctChg'+str(stds)+'Stds'] = pctChg.apply(
        sign_compare, args=(pctChg.std() * stds,))
    ndf[ticker+'PctMA50'] = (c / c.rolling(50).mean())
    ndf[ticker+'PctMA200'] = (c / c.rolling(200).mean())
    ndf[ticker+'RollVol30'] = roll_vol(pctChg, 30)
    for p in rolls: ndf[ticker+'PctChg'+str(p)] = c.pct_change(periods=p)
    # ndf[ticker+'OpenGap20'] = ((o - c1ds) / c1ds).rolling(20).sum()
    # ndf[ticker+'HLDelta20'] = ((h - l) / c1ds).rolling(20).sum()
    ndf[ticker+'Pct52WkH'] = (c / c.rolling(252).max())
    ndf[ticker+'Pct52WkL'] = (c / c.rolling(252).min())
    if not incl_name: ndf['symbol'] = s
    return ndf

# Forward returns
def px_fwd_rets(df, s, periods=[20, 60, 120]):
    ndf = pd.DataFrame()
    for p in periods:
        ndf[s + 'FwdPctChg' + str(p)] = df.pct_change(p).shift(-p)
    return ndf

def px_mom_co_feats(df, ind_df,
                    groups=('Bench', 'Sector','Industry'),
                    rolls=[20,60,120]):
    ndf = pd.DataFrame()
    c,o,l,h,v = df['close'], df['open'], df['low'], df['high'], df['volume']
    bech_idx, sect_idx, ind_idx = groups[0], shorten_name(groups[1]), shorten_name(groups[2])

    for r in rolls:
        ndf['rsBench'+str(r)] = (c.pct_change(r) - ind_df[bech_idx].pct_change(r))
        ndf['rsSect'+str(r)] = (c.pct_change(r) - ind_df[sect_idx].pct_change(r))
        # rsInd meaninful only if len > 10
        ndf['rsInd'+str(r)] = (c.pct_change(r) - ind_df[ind_idx].pct_change(r))

    # vol as a % of 10 and 60 day averages
    ndf['volPctMa10'] = v / v.rolling(10).mean()
    ndf['volPctMa60'] = v / v.rolling(60).mean()

    # of std deviations for benchmark, sector, and industry
    bench_pct_chg = ind_df[bech_idx].pct_change()
    sect_pct_chg = ind_df[sect_idx].pct_change()
    ind_pct_chg = ind_df[ind_idx].pct_change()
    ndf['benchPctChgStds'] = bench_pct_chg.apply(sign_compare, args=(bench_pct_chg.std(),))
    ndf['sectPctChgStds'] = sect_pct_chg.apply(sign_compare, args=(sect_pct_chg.std(),))
    ndf['indPctChgStds'] = ind_pct_chg.apply(sign_compare, args=(ind_pct_chg.std(),))

    ndf['sector'] = groups[1]
    ndf['industry'] = groups[2]

    return ndf

def eq_wgt_indices(profile, px_df, col, group_list, tail=70**2, subset=None):
    names = []
    indices_df = pd.DataFrame()
    for s in group_list:
        idx_ticker = shorten_name(s)
        names.append(idx_ticker)
        symbols = profile[profile[col] == s].symbol.tolist()
#         print('Equal weight index for: %s, %s, %d, %s\n' \
#               % (idx_ticker, s, len(symbols), symbols))
        # if subset: symbols = list(set(symbols).intersection(all_equities))
        index = get_ind_index(px_df[symbols], '1d', tail, idx_ticker)[[idx_ticker]]
        if len(indices_df) == 0:
            indices_df = pd.DataFrame(index)
            continue
        indices_df = pd.concat([indices_df, index], axis=1)
    assert len(names) == len(set(names))
    return indices_df

def max_draw_pull(xs):
    l_dd = np.argmax(np.maximum.accumulate(xs) - xs)
    h_dd = np.argmax(np.array(xs[:l_dd]))
    l_p = np.argmax(xs - np.minimum.accumulate(xs))
    h_p = np.argmin(np.array(xs[:l_p]))
    # drawdown low index, high index; pull low index, high index
    return l_dd, h_dd, l_p, h_p

def get_pct_chg_seasonality(df, rule):
    ss_df = df.pct_change().resample(rule).sum().to_frame()
    ss_df['year'], ss_df['month'] = ss_df.index.year, ss_df.index.month
    ss_df = ss_df.pivot_table(index='year', columns='month').mean()
    ss_pos = [(x, (x+1) if not (x+1) // 12 else 0,
         x+2 if not (x+2) // 12 else x - 10) for x in range(12)]
    return ss_df.loc[('close'),:], ss_pos
