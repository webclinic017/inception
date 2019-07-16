# %%
from utils.basic_utils import *

# %%
# lambdas
list_cols_excl = lambda allCols, exclCols: [x for x in allCols if x not in exclCols]
print_labels = lambda stmt, df: print("{} labels:\n{}".format(stmt, df.columns.tolist()))
valid_cols = lambda df, x: df.columns[df.columns.isin(x)].tolist()
get_FX = lambda FXs, curr: 1 / FXs[FXs.underlyingSymbol == curr].iloc[0].regularMarketPrice
non_empty = lambda df: df[[y for x, y in zip(~df.isna().iloc[0].values, df.columns.tolist()) if x == True]]

# utility functions
def fromts_todate(df, column):
    return [date.fromtimestamp(x) for x in df[column].values]

def if_col_value(df, col_name):
    return df[col_name] if (df.columns.contains(col_name)) else 0

def fs_total_subset(df, columns):
    """Sums a subset of columns"""
    fin_section = df[valid_cols(df, columns)]
    return fin_section.sum(axis=1).sort_index(ascending=True)

def fs_available_labels(df, labels):
    """Returns the labels contained in the DataFrame"""
    return [x for x in labels if df.columns.contains(x)]

def convert_dates(df, dateCols, sortCol):
    for x in dateCols: df[x] = fromts_todate(df, x)
    return df.sort_values(sortCol).set_index(sortCol, drop=True)

def fs_append_ltm(dfA, dfQ, isBs, exclCols):
    """Calculates LTM (sum of last four quarters) and returns appended DataFrame"""
    if len(dfA) > 0 and len(dfQ) > 0 and (dfQ.index[-1] > dfA.index[-1]):
        if not isBs:
            cols_to_add = list_cols_excl(dfQ.columns, exclCols)
            dfQ.loc[dfQ.index[-1], cols_to_add] = dfQ[cols_to_add].sum()
        return dfA.iloc[-3:].append(dfQ.iloc[-1]).sort_index()
    return dfA

def clean_ltm(ds, symbol, filterCol, isBs):
    # appends most recent quarter or last four quarters to historical financials
    df = ds[ds.symbol == symbol]
    df = df.dropna(subset=[filterCol]).sort_index()
    df = fs_append_ltm(df[df.period == 'A'], df[df.period == 'Q'], isBs, excl_cols)
    return df

def cagr_series(val_df):
    years = (val_df.index[-1] - val_df.index[0]).days / 365
    return pow(val_df.iloc[-1] / val_df.iloc[0], (1 / years)) - 1

def get_BS_metrics(symbol):
    # calculate adjusted asset base for valuation
    df = clean_ltm(latest_finBS, symbol, 'cash', True)
    # print_labels("BS", df)
    wc_assets = ['netReceivables', 'inventory']
    wc_liabs = ['accountsPayable']
    total_cash = ['cash', 'shortTermInvestments', 'longTermInvestments']
    total_debt = ['shortLongTermDebt', 'longTermDebt']
    summary = ['totalAssets', 'workingCapital', 'adjAssetBase',
                  'totalCash', 'adjAssetBaseLessCash', 'netDebt']

    df.loc[:, 'workingCapital'] = fs_total_subset(df, wc_assets) - fs_total_subset(df, wc_liabs)
    df.loc[:, 'adjAssetBase'] = df['totalAssets'] + df['workingCapital'].apply(min, args=(0,))
    df.loc[:, 'totalCash'] = fs_total_subset(df, total_cash)
    df.loc[:, 'totalDebt'] = fs_total_subset(df, total_debt)
    df.loc[:, 'netDebt'] = df['totalDebt'] - df['totalCash']
    df.loc[:, 'adjAssetBaseLessCash'] = df['adjAssetBase'] + df['netDebt'].apply(min, args=(0,))

    return (df[valid_cols(df, summary)])

def get_CF_metrics(symbol):
    # calculates steady CF and FCF
    df = latest_finCF[latest_finCF.symbol == symbol]
    df = df.dropna(subset=['totalCashFromOperatingActivities']).sort_index()
    df = fs_append_ltm(df[df.period == 'A'], df[df.period == 'Q'], False, excl_cols)
    # print_labels("CF", df)

    summary = ['totalCashFromOperatingActivities', 'depreciation',
                  'steadyCF', 'capitalExpenditures',
                  'netBorrowings', 'steadyFCF', 'growthCapex',
                  'repurchaseOfStock', 'dividendsPaid', 'sbcAddbacks']

    df.loc[:, 'steadyCF'] = df.totalCashFromOperatingActivities - df['depreciation']
    df.loc[:, 'steadyFCF'] = df['steadyCF'] + df['netBorrowings'].apply(min, args=(0,))
    df.loc[:, 'growthCapex'] = np.abs(df.capitalExpenditures) - df.depreciation
    df.loc[:, 'sbcAddbacks'] = fs_total_subset(
        df, ['changeToNetincome', 'changeToOperatingActivities'])

    return (df[valid_cols(df, summary)])

def get_IS_metrics(symbol, cf_sum):
    # calculates how much cost invested in growth
    df = latest_finIS[latest_finIS.symbol == symbol]
    df = df.dropna(subset=['totalOperatingExpenses']).sort_index()
    df = fs_append_ltm(df[df.period == 'A'], df[df.period == 'Q'], False, excl_cols)
    # print_labels("IS", df)

    cash_cost_growth = ['researchDevelopment',
                        'sellingGeneralAdministrative']
    summary = ['totalRevenue','growthCost']

    cost_gr_df = fs_total_subset(df, cash_cost_growth)
    df.loc[:, 'growthCost'] = (cost_gr_df - cost_gr_df.shift(1)) \
        - cf_sum.sbcAddbacks.apply(min, args=(0,))

    return (df[valid_cols(df, summary)])

def get_val_summary(bs_sum, cf_sum, is_sum):
    # calculate ROA, ROE and asset turnover
    val_df = bs_sum.copy()
    val_df[cf_sum.columns] = cf_sum
    val_df[is_sum.columns] = is_sum

    val_df['avgTotalAssets'] = val_df['totalAssets'].rolling(2).mean() # avg asset base
    val_df['avgAssetBaseLC'] = val_df['adjAssetBaseLessCash'].rolling(2).mean() # avg asset base
    val_df.avgTotalAssets[0] = val_df.totalAssets[0]
    val_df.avgAssetBaseLC[0] = val_df.adjAssetBaseLessCash[0]
    val_df['reinvCapital'] = (val_df['growthCapex'] + val_df['growthCost']).apply(max, args=(0,))
    val_df['ROA'] = (val_df.steadyCF / val_df.avgTotalAssets)
    val_df['ROE'] = (val_df.steadyFCF / val_df.avgAssetBaseLC)
    val_df['AT'] = (val_df.totalRevenue / val_df.avgTotalAssets)
    val_df['AAT'] = (val_df.totalRevenue / val_df.avgAssetBaseLC)

    return val_df

def value_equity(symbol):
    # performs basic intrinsic value calculations
    waterfall = {}
    waterfall['symbol'] = symbol
    waterfall['storeDate'] = datetime.strptime(str(today_date), '%Y-%m-%d').timestamp()

    fin_stats_df = latest_finstats[latest_finstats.symbol == symbol]
    quote_df = quotes[quotes.symbol == symbol]

    # PENDING: need to add symbol and storeDate so we can save as an aggregate dataset
    bs_sum = get_BS_metrics(symbol) / div_unit
    cf_sum = get_CF_metrics(symbol) / div_unit
    is_sum = get_IS_metrics(symbol, cf_sum) / div_unit
    val_df = get_val_summary(bs_sum, cf_sum, is_sum)
    val_df['symbol'] = symbol
    val_df['storeDate'] = datetime.strptime(str(today_date), '%Y-%m-%d').timestamp()

    # historical growth rates
    ROIC_stats = val_df[['ROA', 'ROE','AT', 'AAT']].mean()
    ROE = ROIC_stats.ROE

    # Calculate valuation multiples
    mult = round( 1 / discount_rate, 2)
    growth_rate = max(cagr_series(val_df[growth_cols]).median(), 0)
    gr_mult = mult + (growth_rate * growth_cap_factor * 100)

    # save assumptions during daily computations
    names = [
        'baseRate', 'equityRiskPremium', 'projFedFundsIncr', 
        'discountRate', 'growthRate']
    values = [
        base_rate, risk_premium, proj_increase, 
        discount_rate, growth_rate]
    waterfall.update({x: y for x, y in zip(names, values)})
    waterfall.update(ROIC_stats.to_dict())

    # calculate intrinsic value of equity    
    steadyCF = val_df.steadyCF[-1]
    npv_CF = steadyCF * mult
    names = ['steadyCF', 'baseMult', 'npv_CF',]
    values = [steadyCF, mult, npv_CF]
    waterfall.update({x: y for x, y in zip(names, values)})    
    totalreinvCapital = val_df.reinvCapital.sum()
    projCashROE = totalreinvCapital * ROE
    npv_GR = projCashROE * gr_mult if totalreinvCapital > 0 else 0
    names = ['totalReinvCapital', 'projCashROE', 'growthMult', 'npvGrowthCF']
    values = [totalreinvCapital, projCashROE, gr_mult, npv_GR]
    waterfall.update({x: y for x, y in zip(names, values)})

    # FX adjustment
    curr = fin_stats_df.iloc[0].financialCurrency
    FX = 1 if curr == 'USD' else get_FX(FXs, curr)

    # given dual class shares we equity value only, not value per share (manual)
    values = [x * FX for x in [npv_CF, -val_df.netDebt[-1], npv_GR]]
    equityVal = sum([values[0], values[1], values[2]])
    values.append(equityVal)
    currentVal = quote_df.iloc[0].marketCap / div_unit
    prem_disc = currentVal / equityVal
    values.append(currentVal)
    values.append(prem_disc)
    values.append(curr)
    values.append(FX)
    names = ['npvSteadyCF', 'netDebt', 'npvGrowth', 'equityValue', 'currentValue', 'premDisc', 'financialCurrency', 'FX']
    waterfall.update({x: y for x, y in zip(names, values)})

    return waterfall, val_df

def create_IV_ds():
    eqty_symbols = quotes[quotes.quoteType == 'EQUITY'].symbol.unique().tolist()
    # excl_list = ['TSM'] # one-time exclude because missing
    # eqty_symbols = [x for x in eqty_symbols if x not in excl_list]
    # full loop value each company
    waterfalls = pd.DataFrame() # key assumptions
    val_sheets = pd.DataFrame() # valuation datasets for each co
    for symbol in eqty_symbols:
        try:
            print('Intrinsic value for {}'.format(symbol))
            waterfall, val_df = value_equity(symbol)
            val_sheets = val_sheets.append(val_df)
            waterfalls = waterfalls.append(waterfall, ignore_index=True)
        except Exception as e: 
            print(e)
    # waterfalls.set_index('symbol', inplace=True)

    # save the results to S3
    csv_store(waterfalls, 'valuation/waterfall/', csv_ext.format(str(today_date)))
    csv_store(val_sheets, 'valuation/backup/', csv_ext.format(str(today_date)))

# %%
UNIT_SCALE = 10**9
date_cols = ['endDate', 'storeDate']
excl_cols = ['maxAge', 'symbol', 'period', 'storeDate']
rates = ["^IRX", "^FVX", "^TNX", "^TYX"]
key_cols = [
    'totalAssets','adjAssetBaseLessCash', 'netDebt', 'totalRevenue',
    'steadyCF', 'steadyFCF', 'reinvCapital', 'ROA', 'ROE', 'AT', 'AAT']
growth_cols = [
    'totalRevenue','totalAssets', 'totalCashFromOperatingActivities',
    'steadyCF', 'steadyFCF', 'capitalExpenditures', 'reinvCapital']
# growth_cols = ['steadyFCF']

dates = read_dates('quote')
tgt_date = [dates[-1]] # hardcoded for now
quotes = load_csvs('quote_consol', tgt_date)
quotes.set_index('symbol', drop=False, inplace=True)
profile = load_csvs('summary_detail', ['assetProfile'])
latest_keystats = load_csvs('summary_detail', ['defaultKeyStatistics/' + str(tgt_date[0])])
latest_finstats = load_csvs('summary_detail', ['financialData/' + str(tgt_date[0])])

FXs = quotes[(quotes.quoteType == 'CURRENCY')]
FXs.underlyingSymbol = [x.split('=X')[0].split('USD')[0] for x in FXs.symbol.unique().tolist()]
get_FX = lambda FXs, curr: 1 / FXs[FXs.underlyingSymbol == curr].iloc[0].regularMarketPrice

# assumptions, THIS SHOULD COME FROM EXTERNAL CONFIG FILE
base_rate_symbol = "^TNX"
base_rate = quotes.loc[base_rate_symbol,:].regularMarketPrice / 100
risk_premium = 500 / 10000 # pending set up externally
proj_increase = 0 # pending set up externally
growth_cap_factor = 0.3 # pending set up externally
growth_discount_premium = 1 # pending set up externally
discount_rate = base_rate + risk_premium + proj_increase
div_unit = 10**9

# load all financial files
latest_finBS = load_csvs('summary_detail', ['financials-BS'])
latest_finIS = load_csvs('summary_detail', ['financials-IS'])
latest_finCF = load_csvs('summary_detail', ['financials-CF'])

# convert timestamps to dates
latest_finBS = convert_dates(latest_finBS, date_cols, 'endDate')
latest_finIS = convert_dates(latest_finIS, date_cols, 'endDate')
latest_finCF = convert_dates(latest_finCF, date_cols, 'endDate')

# %%
iv_waterfall = pd.read_csv(csv_load(f'valuation/waterfall/{str(today_date)}'), parse_dates=True, index_col='symbol')
pred_df = pd.read_csv(csv_load(f'recommend/equities/{str(today_date)}'), parse_dates=True, index_col='symbol')
long_symbols, short_symbols = pred_df.loc[pred_df.shares > 0].index, pred_df.loc[pred_df.shares < 0].index

# %%
symbol = 'SQ'
waterfall, val_df = value_equity(symbol)
waterfall
# val_df.T

# %%
# iv_waterfall.premDisc.describe()
iv_waterfall.loc[long_symbols[:5], 'premDisc']
# iv_waterfall.reindex(index=long_symbols).premDisc.describe(percentiles=[.01,.25,.5,.75,.99])

# %% testing random stuff
# latest_finIS.columns
# str(today_date)

# %%
if __name__ == '__main__':
    create_IV_ds()