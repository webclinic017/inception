import logging, json, os, sys
from datetime import datetime, date
import pandas as pd
from utils.basic_utils import store_s3, csv_store, csv_ext
from ib_insync import IB, util
from ib_insync import Contract, ContractDetails, ContractDescription
from ib_insync import Stock, Forex, Index, Future, ContFuture, CFD, Option, Bond
from ib_insync.client import Client
from ib_insync.objects import FundamentalRatios

# util.logToConsole(logging.WARNING)
# logger = logging.getLogger()
# logger.setLevel("ERROR")

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

get_hist_data_type = lambda x: wts_dict[x] if x in wts_dict.keys() else 'ADJUSTED_LAST'

def get_hist_data(ib, contract, duration, bar_size, hist_data_type):
    " Retrieves historical price and volume for a given contract "
    # takes too long: ~30s per instrument
    # see: https://groups.io/g/insync/topic/reqhistoricaldata_is_too_slow/6011327?p=,,,20,0,0,0::recentpostdate%2Fsticky,,,20,1,40,6011327
    print(f'Ingesting historical pricing for {contract.symbol}')
    bars = ib.reqHistoricalData(
        contract, endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size, 
        whatToShow=hist_data_type, 
        useRTH=True)
    return util.df(bars)

def ingest_contract_details(ib, ib_config, sel_contracts):
    print("Ingesting contract details...")
    store_path = ib_config['store_path']['contracts']
    s_l = []
    for contract in sel_contracts:
        cds = ib.reqContractDetails(contract)
        sec_contracts = [cd.contract for cd in cds]
        df = util.df(sec_contracts)
        print(f'{contract.symbol} details: {df is not None}')
        if df is not None: s_l.append(df)
    df = pd.concat(s_l, axis=0)
    csv_store(df, store_path, f'contract_details.csv')

def ingest_historical_pricing(ib, ib_config, sel_contracts):
    print("Ingesting historical price and volume...")
    store_path = ib_config['store_path']['price_vol']
    for contract in sel_contracts:
        c_type = contract.__class__.__name__
        hist_data_type = get_hist_data_type(c_type)
        df = get_hist_data(ib, contract, duration, bar_size, hist_data_type)
        print(f'{contract.symbol} pricing: {df is not None}')
        csv_store(df, store_path, csv_ext.format(f'{contract.symbol}'))

def ingest_fundamental_ratios(ib, ib_config, sel_contracts):
    print("Ingesting fundamental ratios...")
    store_path = ib_config['store_path']['fundamental_ratios']
    s_l, idx_l = [], []
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


def ingest_fundamental_reports(ib, ib_config, contracts):
    print("Ingesting fundamental reports...")
    store_path = ib_config['store_path']['fundamental_reports']
    for contract in contracts:
        # fundamental reports in XML
        fund_reports = ['ReportsFinSummary', 'ReportSnapshot', 'ReportsFinStatements', 'RESC']
        for r in fund_reports:
            fr = ib.reqFundamentalData(contract, r)
            path = f'{store_path.format(r)}{contract.symbol}.xml'
            print(path)
            store_s3(fr, path)

if __name__ == '__main__':
    
    hook = sys.argv[1]
    store_date = str(date.today())

    ib_sleep = 5
    duration = '5 Y'
    bar_size = '1 day'
    wts_dict = {'Forex': 'BID_ASK'}
    ib_univ = pd.read_csv('./utils/ib_universe.csv', index_col='symbol')
    
    ib = IB()
    ib.connect('127.0.0.1', 7496, clientId=1)

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
    ss_df = ib_univ.loc[ib_univ.type.isin(['Option'])]
    options = [Option(t, exchange=ss_df.loc[t, "exchange"], currency=ss_df.loc[t, "currency"]) for t in list(ss_df.index)]

    subset = {
        'Stock': stocks,
        'ETF': etfs,
        'Index': indices,
        'Forex': forex,
        'ContFuture': futures,
        # 'Option': options,
    }

    if hook == 'details':
        sel_contracts = []
        for k in subset.keys(): sel_contracts.extend(subset[k])
        ib.qualifyContracts(*sel_contracts)
        # ingest_contract_details(ib, ib_config, sel_contracts)
        ingest_contract_details(ib, ib_config, sel_contracts)
    elif hook == 'pricing':
        sample = sys.argv[2]
        sel_contracts = subset[sample]
        ib.qualifyContracts(*sel_contracts)
        ingest_historical_pricing(ib, ib_config, sel_contracts)
    elif hook == 'fundamental_ratios':
        sample = sys.argv[2]
        sel_contracts = subset[sample]
        ib.qualifyContracts(*sel_contracts)
        ingest_fundamental_ratios(ib, ib_config, sel_contracts)
    elif hook == 'fundamental_reports':
        sample = sys.argv[2]
        sel_contracts = subset[sample]
        ib.qualifyContracts(*sel_contracts)
        ingest_fundamental_reports(ib, ib_config, sel_contracts)
    else:
        print('Please enter a valid option:')
        print('details, pricing, fundamental_ratios, or fundamental_reports')
        print(f'pricing, ratios, and reports require a subset from: {subset.keys()}')

    ib.disconnect()