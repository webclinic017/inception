import logging, json, os
from datetime import datetime
import pandas as pd
from utils.basic_utils import store_s3, csv_store, csv_ext
from ib_insync import IB, util
from ib_insync import Contract, ContractDetails, ContractDescription
from ib_insync import Stock, Forex, Index, Future, ContFuture, CFD, Option, Bond
from ib_insync.client import Client
from ib_insync.objects import FundamentalRatios

util.logToConsole(logging.WARNING)
logger = logging.getLogger()
logger.setLevel("ERROR")

"""
ib_ds
    contracts (overriden)
    historical (daily)
        price_vol
        options
    fundamental
        ratios (daily)
        reports (override)
            'ReportsFinSummary' (override)
            'ReportSnapshot' (override)
            'ReportsFinStatements' (override)
            'RESC' (override)
    positions (daily)
"""

ib_config = {
    'store_path': {
        'contracts': 'ib_ds/',
        'price_vol': 'ib_ds/historical/price_vol/',
        'fundamental_ratios': 'ib_ds/fundamental/ratios/',
        'fundamental_reports': 'ib_ds/fundamental/reports/{}/',
        'options': 'ib_ds/historical/options/',
        'positions': 'ibk_positions/'
    }
}

ib_sleep = 5
tmp_folder = './tmp/'
os.makedirs(tmp_folder, exist_ok=True)

ib_univ = pd.read_csv('./utils/ib_universe.csv', index_col='symbol')
store_date = datetime.now().strftime('%Y-%m-%d')
duration = '5 Y'
bar_size = '1 day'
wts_dict = {'Forex': 'BID_ASK'}

get_hist_data_type = lambda x: wts_dict[x] if x in wts_dict.keys() else 'ADJUSTED_LAST'

def get_hist_data(contract, duration, bar_size, hist_data_type):
    " Retrieves historical price and volume for a given contract "
    print(f'Historical data for {contract}')
    bars = ib.reqHistoricalData(
        contract, endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size, 
        whatToShow=hist_data_type, 
        useRTH=True)
    return util.df(bars)

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)

contracts = []

ss_df = ib_univ.loc[ib_univ.type.isin(['Stock'])]
stocks = [Stock(t, exchange=ss_df.loc[t, "exchange"], currency=ss_df.loc[t, "currency"]) for t in list(ss_df.index)]
ss_df = ib_univ.loc[ib_univ.type.isin(['ETF'])]
etfs = [Stock(t, exchange=ss_df.loc[t, "exchange"], currency=ss_df.loc[t, "currency"]) for t in list(ss_df.index)]
ss_df = ib_univ.loc[ib_univ.type.isin(['Index'])]
indices = [Index(t, exchange=ss_df.loc[t, "exchange"], currency=ss_df.loc[t, "currency"]) for t in list(ss_df.index)]
ss_df = ib_univ.loc[ib_univ.type.isin(['Forex'])]
forex = [Forex(t) for t in list(ss_df.index)]
ss_df = ib_univ.loc[ib_univ.type.isin(['ContFuture'])]
futures = [ContFuture(t, exchange=ss_df.loc[t, "exchange"], currency=ss_df.loc[t, "currency"]) for t in list(ss_df.index)]
# ss_df = ib_univ.loc[ib_univ.type.isin(['Option'])]
# options = [Option(t, exchange=ss_df.loc[t, "exchange"], currency=ss_df.loc[t, "currency"]) for t in list(ss_df.index)]

contracts = stocks + etfs + indices + forex + futures
ib.qualifyContracts(*contracts)

"""
print("Contract details...")
store_path = ib_config['store_path']['contracts']
s_l = []
for contract in contracts:
    cds = ib.reqContractDetails(contract)
    sec_contracts = [cd.contract for cd in cds]
    df = util.df(sec_contracts)
    print(f'{contract.symbol} details: {df is not None}')
    s_l.append(df)
df = pd.concat(s_l, axis=0)
csv_store(df, store_path, f'contract_details.csv')
"""
"""
print("Ingesting historical price and volume...")
store_path = ib_config['store_path']['price_vol']
sel_contracts = stocks
for contract in sel_contracts:
    c_type = contract.__class__.__name__
    hist_data_type = get_hist_data_type(c_type)
    df = get_hist_data(contract, duration, bar_size, hist_data_type)
    print(f'{contract.symbol} pricing: {df is not None}')
    csv_store(df, store_path, csv_ext.format(f'{contract.symbol}'))
    break
"""
"""
print("Ingesting fundamental ratios...")
store_path = ib_config['store_path']['fundamental_ratios']
s_l, idx_l = [], []
sel_contracts = stocks + etfs
for contract in sel_contracts:
    ticker = ib.reqMktData(contract, '258')
    ib.sleep(ib_sleep)
    fr = ticker.fundamentalRatios
    print(f'{contract.symbol} ratios: {fr is not None}')
    if fr is not None: 
        idx_l.append(contract.symbol)
        s_l.append(fr)
df = util.df(s_l)
df['symbol'] = idx_l
df['storeDate'] = store_date
csv_store(df, store_path, f'{store_date}.csv')
print(df)
"""
"""
print("Ingesting fundamental reports...")
store_path = ib_config['store_path']['fundamental_reports']
for contract in stocks:
    # fundamental reports in XML
    fund_reports = ['ReportsFinSummary', 'ReportSnapshot', 'ReportsFinStatements', 'RESC']
    for r in fund_reports:
        fr = ib.reqFundamentalData(contract, r)
        path = f'{store_path.format(r)}{contract.symbol}.xml'
        print(path)
        store_s3(fr, path)

logger.info("Equities Complete")
"""
ib.disconnect()