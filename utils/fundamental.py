# imports
from utils.basic_utils import *
from utils.pricing import *
from utils.imports import *
from utils.structured import *

# utility functions
numeric_cols = lambda df: list(df.columns[df.dtypes.apply(is_numeric_dtype).values])
col_mapper = lambda pre, fields, period: {x:'_'.join([pre,x,period]) for x in fields}
filter_cols = lambda columns, c: [x for x in columns if c in x]
day_delta = lambda df, d: df - df.shift(d)

def preproc_df(df, key, pipe, context):
    proc_df = df.copy()
    if key in pipe:
        for fn in pipe[key]: proc_df = fn(proc_df, context)
    return proc_df

def preproc_outlier(df, context):
    """ Remove rows where values > treshold """
    ds_dict = context['ds_dict']
    treshold = ds_dict['outlier']
    re_idx_df = df.reindex().reset_index()
    fields = numeric_cols(re_idx_df)
    ol_locs = list(
        re_idx_df[
            (np.abs(re_idx_df[fields]) > treshold)\
            .any(1)].index)
    keep_ix = np.isin(np.arange(len(df)), ol_locs)
    return df.loc[~keep_ix, :]

def get_daily_ts(key, ds_dict, dates):
    index_col = ds_dict[key]['index']
    features = ds_dict[key]['features']
    df = load_csvs(path, dates)
#     df.loc[:, index_col[0]] = pd.to_datetime(df[index_col[0]], unit='s')
    df.loc[:, index_col] = pd.to_datetime(df[index_col], unit='s')
    df.set_index(index_col, drop=True, inplace=True)
    df.index.set_names(index_col, inplace=True)
    return df[features]

def df_wide_transform(df, context):
    periods = context['periods']
    pre = context['pre']
    # applies only to numeric fields
    fields = numeric_cols(df)
    super_list = []
    for p in periods:
        filered_df = df[df.period == p][fields]
        filered_df = filered_df.rename(columns=col_mapper(pre, fields, p))
        super_list.append(filered_df)
    new_df = pd.concat(super_list, axis=1, sort=True).dropna(axis=1)
    new_df['symbol'] = df.symbol.iloc[0]
    return new_df

# pre-processing functions
def preproc_perc_price(df, context):
    ds_dict = context['ds_dict']
    perc_price_cols = ds_dict['perc_price']
    hist_price_df = context['hist_price']
#     df.loc[:, perc_price_cols] = df[perc_price_cols] / hist_price_df.values
    df.loc[:, perc_price_cols] = df[perc_price_cols].div(hist_price_df.values, axis=0)
    return df

def preproc_divide(df, context):
    ds_dict = context['ds_dict']
    divide_tuple = ds_dict['divide']
    df.loc[:, divide_tuple[1]] =  df[divide_tuple[1]].div(df[divide_tuple[0]], axis=0)
    df.drop(columns=divide_tuple[0], inplace=True)
    return df

def preproc_scale(df, context):
    ds_dict = context['ds_dict']
    scale_cols = ds_dict['scale']
    # for growth we substract from prior date
    df.loc[:, scale_cols] /= df[scale_cols].mean()
    return df

def preproc_eps_estimates(df, context):
    fields = numeric_cols(df)
    growth_cols = filter_cols(fields, 'growth')
    other_cols = excl(fields, growth_cols)
    # for growth we substract from prior date
    df.loc[:, growth_cols] = day_delta(df[growth_cols], 1)
    # shows when there are jumps on low, avg, and high estimates
    df.loc[:, other_cols] = df[other_cols].pct_change()
    return df

def preproc_day_delta(df, context): return day_delta(df[numeric_cols(df)], 1)

def preproc_eps_trend(df, context):
    super_list = []
    periods = context['periods']
    fields = numeric_cols(df)
    growth_cols = filter_cols(fields, 'growth')
    # for growth we substract from prior date
    df.loc[:, growth_cols] = day_delta(df[growth_cols], 1)
    for p in periods:
        cols = filter_cols(fields, p)
        slope_cols = list(filter(lambda x, y='growth': y not in x, cols))
        # divide current, -7, -30, and -60 days by -90 days ago for trend
        # then substract from prior day, for signal changes
        df.loc[:, slope_cols] = day_delta(df[slope_cols].div(df[slope_cols[-1]], axis=0), 1)
    return df
