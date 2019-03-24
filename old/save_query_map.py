from basic_utils import *

url_key = 'url'
enc_key = 'enc_key'
enc_val = 'enc_val'


query_map = {
    'summary': {
        url_key:'https://query1.finance.yahoo.com/v10/finance/quoteSummary/{0}?formatted=true&lang=en-US&region=US&{1}&corsDomain=finance.yahoo.com',
        enc_key: 'modules',
        enc_val:'defaultKeyStatistics,assetProfile,financialData,balanceSheetHistory,balanceSheetHistoryQuarterly,cashflowStatementHistory,cashflowStatementHistoryQuarterly,incomeStatementHistory,incomeStatementHistoryQuarterly,calendarEvents,earnings,earningsHistory,earningsTrend,recommendationTrend,upgradeDowngradeHistory,indexTrend,fundOwnership,insiderHolders,institutionOwnership,majorDirectHolders,majorHoldersBreakdown,netSharePurchaseActivity'
    },
    'option': {
        url_key:'https://query1.finance.yahoo.com/v7/finance/options/{0}?formatted=true&lang=en-US&region=US&straddle=false&{1}&corsDomain=finance.yahoo.com',
        enc_key: 'date'
    },
    'quote':{
        url_key:'https://query1.finance.yahoo.com/v7/finance/quote?formatted=true&lang=en-US&region=US&{0}&corsDomain=finance.yahoo.com',
        enc_key: 'symbols'
    }
}

def save_config(config, fname):
    with open(fname, 'w') as file:
        data = json.dumps(config, indent=1)
        file.write(data)
        file.close()
        print('Saving', fname)

save_config(query_map, 'query_map.json')
