import datetime, logging, json, os
import pandas as pd
from utils.basic_utils import store_s3, csv_store
from ib_insync import IB, util
from ib_insync import Contract, ContractDetails, ContractDescription
from ib_insync import Stock, Forex, Index, Future, ContFuture, CFD, Option, Bond
from ib_insync.client import Client
from ib_insync.objects import FundamentalRatios

"""
Folder structure:
ib_datasets
    xdetails (overriden)
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

util.logToConsole(logging.WARNING)
logger = logging.getLogger()
logger.setLevel("ERROR")

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)
ib_sleep = 0.5

_folder = './tmp/'
os.makedirs(_folder, exist_ok=True)

universe = pd.read_csv('./utils/ib_universe.csv', index_col='symbol')
types = list(universe.type.unique())
all_contracts = []
wts_dict = {'FX': 'BID_ASK'}
duration = '15 Y'
bar_size = '1 d'

active_type = 'Stock'
hist_data_type = wts_dict[active_type] if active_type in wts_dict.keys() else 'ADJUSTED_LAST'
contracts = []
symbols = universe.loc[universe.type.isin([active_type])]
contracts = [Stock(s, 
        universe.loc[s, 'exchange'], 
        universe.loc[s, 'currency']) 
        for s in symbols]
all_contracts.extend(contracts)

def get_historical(contracts, duration, bar_size, hist_data_type, store_path):
    for contract in contracts:
        bars = ib.reqHistoricalData(
            contract, endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size, 
            whatToShow=hist_data_type, 
            useRTH=True)
        df = util.df(bars)
        df.to_csv(f'{_folder}{contract.symbol}.csv')

# CONTRACT DETAILS
"""
sample_contracts = [
    Contract(conId=270639),
    Stock('TSLA', exchange='SMART', currency='USD'),
    Index('VIX'),
    Stock('SPY', exchange='SMART', currency='USD'),    
    Forex('EURUSD'),
    ContFuture('HG', exchange='NYMEX', currency='USD'),
    Option('FXI', exchange='SMART', currency='USD'),
]
s_l = []
for contract in sample_contracts:
    cds = ib.reqContractDetails(contract)
    sec_contracts = [cd.contract for cd in cds]
    df = util.df(sec_contracts)
    s_l.append(df)
pd.concat(s_l, axis=0).to_csv(f'{_folder}contract-details.csv')
"""

# EQUITIES
"""
logger.info("Equities Start")
logger.info("Contracts...")
categ_list = pd.read_csv('company_map.csv', index_col=[0])
c_list = list(categ_list.index.unique())
contracts = [
    Stock(t, 
    categ_list.loc[t, 'exchange'], 
    categ_list.loc[t, 'currency']) 
for t in c_list]

# cds = ib.reqContractDetails(Stock(c_list[0]))
# contracts = [cd.contract for cd in cds]
# print({key: set(getattr(c, key) for c in contracts) for key in Contract.defaults})
"""

"""
logger.info("Ingesting historical price and volume...")
for contract in contracts:
    bars = ib.reqHistoricalData(
        contract, 
        endDateTime='', 
        durationStr='15 Y',
        barSizeSetting='1 day', 
        whatToShow='ADJUSTED_LAST', 
        useRTH=True)
    df = util.df(bars)
    df.to_csv(f'{_folder}{contract.symbol}.csv')
    break
"""
"""
logger.info("Ingesting fundamental ratios...")
s_l, idx_l = [], []
for contract in contracts:
    ticker = ib.reqMktData(contract, '258')
    ib.sleep(ib_sleep)
    fr = ticker.fundamentalRatios
    idx_l.append(contract.symbol)
    s_l.append(fr)
    # with open(f'{_folder}{contract.symbol}.json', 'w') as f:
    #     f.write(json.dumps(vars(fr)))
df = util.df(s_l)
df.index = idx_l
df.to_csv(f'{_folder}fundamental-ratios.csv')
# """
"""
logger.info("Ingesting fundamental reports...")
for contract in contracts:
    # fundamental reports in XML
    fund_reports = ['ReportsFinSummary', 'ReportSnapshot', 'ReportsFinStatements', 'RESC']
    for r in fund_reports:
        fr = ib.reqFundamentalData(contract, r)
        with open(f'{_folder}{contract.symbol}{r}.xml', 'w') as f:
            f.write(fr)
logger.info("Equities Complete")
"""

# INDICES
"""
categ_list = pd.read_csv('index_request.csv', index_col=[0])
c_list = list(categ_list.index.unique())
contracts = [
    Index(t, categ_list.loc[t, 'exchange'], categ_list.loc[t, 'currency']) for t in c_list]
ib.qualifyContracts(*contracts)

# historical price and volume (OCHLV)
for contract in contracts:
    bars = ib.reqHistoricalData(
        contract, 
        endDateTime='', 
        durationStr='15 Y',
        barSizeSetting='1 day', 
        whatToShow='ADJUSTED_LAST', 
        useRTH=True)
    df = util.df(bars)
    df.to_csv(f'{_folder}{contract.symbol}.csv')
"""

# ETF
"""
categ_list = pd.read_csv('etf_request.csv', index_col=[0])
c_list = list(categ_list.index.unique())
contracts = [
    Stock(t, categ_list.loc[t, 'exchange'], categ_list.loc[t, 'currency']) for t in c_list]
ib.qualifyContracts(*contracts)

# historical price and volume (OCHLV)
for contract in contracts:
    bars = ib.reqHistoricalData(
        contract, 
        endDateTime='', 
        durationStr='15 Y',
        barSizeSetting='1 day', 
        whatToShow='ADJUSTED_LAST', 
        useRTH=True)
    df = util.df(bars)
    df.to_csv(f'{_folder}{contract.symbol}.csv')
    ib.sleep(1)
"""

# FUTURES
"""
categ_list = pd.read_csv('future_map.csv', index_col=[0], delimiter=',')
c_list = list(categ_list.index.unique())
contracts = [
    ContFuture(t, 
        exchange=categ_list.loc[t, 'exchange'],
        currency=categ_list.loc[t, 'currency'],
    ) for t in c_list
]
ib.qualifyContracts(*contracts)

for contract in contracts:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='15 Y',
        barSizeSetting='1 day',
        whatToShow='ADJUSTED_LAST',
        useRTH=True)
    df = util.df(bars)
    df.to_csv(f'{_folder}{contract.symbol}.csv')
"""

# FOREX
"""
categ_list = pd.read_csv('fx_request.csv', index_col=[0], delimiter=',')
c_list = ['EURUSD', 'GBPUSD', 'USDHKD', 'USDJPY']
contracts = [Forex(t) for t in c_list]
ib.qualifyContracts(*contracts)

# historical price and volume (OCHLV)
for contract in contracts:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='15 Y',
        barSizeSetting='1 day',
        whatToShow='BID_ASK',
        useRTH=True)
    df = util.df(bars)
    df.to_csv(f'{_folder}{contract.localSymbol}.csv')
"""

ib.disconnect()