from basic_utils import *

dates = read_dates('summary')
symbol_col = 'symbol'

# lambdas
show_structure = lambda dict_struct: {k: type(v) for k, v in dict_struct.items()}
remove_empty_keys = lambda dict_struct: {k: v for k, v in dict_struct.items() if dict_struct[k]}
get_column_order = lambda route: list(remove_empty_keys(route))
get_symbol_index = lambda route, indexKey: [indexKey for x in range(len(route))]

def create_df(route, indexKey):
    order = get_column_order(route[-1])
    df = pd.DataFrame(route)[order]
    df = set_symbol(df, indexKey)
    return df
def create_normalized_df(route, indexKey):
    order = get_column_order(route[-1])
    df = clean_up_fmt(json_normalize(route))[order]
    df = set_symbol(df, indexKey)
    return df
def clean_single_row_df(route):
    order = get_column_order(route)
    df = clean_up_fmt(json_normalize(route))[order]
    return df
def single_row_df(route, indexKey):
    df = clean_single_row_df(route)
    df = set_symbol(df, indexKey)
    return df

def set_storeDate(df, date):
    df['storeDate'] = datetime.strptime(str(date), '%Y-%m-%d').timestamp()
    return df
def set_symbol(df, symbol):
    df[symbol_col] = symbol
    return df

fin_stmt_mappings = {
    "CF":{"A":"cashflowStatementHistory",
        "Q":"cashflowStatementHistoryQuarterly",
        "B":"cashflowStatements"},
    "BS":{"A":"balanceSheetHistory",
        "Q":"balanceSheetHistoryQuarterly",
        "B":"balanceSheetStatements"},
    "IS":{"A":"incomeStatementHistory",
        "Q":"incomeStatementHistoryQuarterly",
        "B":"incomeStatementHistory"}    }
def parse_finstmt(summary, stmt, symbol):
    df = pd.DataFrame()
    mapping = fin_stmt_mappings[stmt]
    A = summary[mapping['A']][mapping['B']]
    if A:
        df = create_normalized_df(A, symbol)
        df['period'] = 'A'
    Q = summary[mapping['Q']][mapping['B']]
    if Q:
        q_df = create_normalized_df(Q, symbol)
        q_df['period'] = 'Q'
        df = df.append(q_df, sort=False)
    return df

def parse_earnings_trend(summary, symbol):
    route = summary['earningsTrend']['trend']
    epsEst_df = pd.DataFrame()
    revEst_df = pd.DataFrame()
    epsTrend_df = pd.DataFrame()
    epsRev_df = pd.DataFrame()
    period_df = pd.DataFrame()

    for item in route:
        epsEst_df = epsEst_df.append(single_row_df(item.pop('earningsEstimate'), symbol), sort=False)
        revEst_df = revEst_df.append(single_row_df(item.pop('revenueEstimate'), symbol), sort=False)
        epsTrend_df = epsTrend_df.append(single_row_df(item.pop('epsTrend'), symbol), sort=False)
        epsRev_df = epsRev_df.append(single_row_df(item.pop('epsRevisions'), symbol), sort=False)
        period_df = period_df.append(clean_single_row_df(item), sort=False)

    if 'growth' in epsEst_df.columns: epsEst_df.drop(labels='growth', axis=1, inplace=True)
    if 'growth' in revEst_df.columns: revEst_df.drop(labels='growth', axis=1, inplace=True)

    epsEst_df = pd.concat([period_df, epsEst_df], axis=1)
    revEst_df = pd.concat([period_df, revEst_df], axis=1)
    epsTrend_df = pd.concat([period_df, epsTrend_df], axis=1)
    epsRev_df = pd.concat([period_df, epsRev_df], axis=1)

    return epsEst_df, revEst_df, epsTrend_df, epsRev_df


# In[28]:


def get_mult_rows(key, summary, symbol):
    if key in summary: return create_normalized_df(summary[key], symbol)
def get_single_row(key, summary, symbol):
    if key in summary: return single_row_df(summary[key], symbol)

def direct_row(summary, symbol): return single_row_df(summary, symbol)
def direct_rows(summary, symbol): return create_normalized_df(summary, symbol)

def unpack_summaries(dates):
    # unpack daily summary JSON files
    for d in dates:

        profile_df = pd.DataFrame()
        officers_df = pd.DataFrame()
        keyStats_df = pd.DataFrame()
        finStats_df = pd.DataFrame()
        finStmtBS_df = pd.DataFrame()
        finStmtIS_df = pd.DataFrame()
        finStmtCF_df = pd.DataFrame()
        earningsEst_df = pd.DataFrame()
        revenueEst_df = pd.DataFrame()
        epsTrend_df = pd.DataFrame()
        epsRevisions_df = pd.DataFrame()
        netSharePA_df = pd.DataFrame()
        majorHolders_df = pd.DataFrame()
        ownershipList_df = pd.DataFrame()
        fundOwnership_df = pd.DataFrame()
        recommendHistory_df = pd.DataFrame()
        recommendTrend_df = pd.DataFrame()
        insiderHolders_df = pd.DataFrame()
        earningsHistory_df = pd.DataFrame()
        indexTrend_df = pd.DataFrame()

        print('Unpacking summary for {}'.format(d))
        fileList = list_files('summary', d)
        i = 0
        for f in fileList:
            symbol = f.split('/')[2].split('.json')[0]
            consol_summary = json_load(f)

            if consol_summary:
                summary = consol_summary[0]

                # profile
                key = 'assetProfile'
                if key in summary:
                    officers = summary[key].pop('companyOfficers')
                    if officers:
                        officers_df = officers_df.append(direct_rows(officers, symbol), sort=False)
                    profile_df = profile_df.append(get_single_row(key, summary, symbol), sort=False)

                # financials
                A, Q = fin_stmt_mappings['CF']['A'], fin_stmt_mappings['CF']['Q']
                if A in summary and Q in summary:
                    finStmtCF_df = finStmtCF_df.append(parse_finstmt(summary, 'CF', symbol), sort=False)
                A, Q = fin_stmt_mappings['BS']['A'], fin_stmt_mappings['BS']['Q']
                if A in summary and Q in summary:
                    finStmtBS_df = finStmtBS_df.append(parse_finstmt(summary, 'BS', symbol), sort=False)
                A, Q = fin_stmt_mappings['IS']['A'], fin_stmt_mappings['IS']['Q']
                if A in summary and Q in summary:
                    finStmtIS_df = finStmtIS_df.append(parse_finstmt(summary, 'IS', symbol), sort=False)

                # other datasets
                key = 'defaultKeyStatistics'
                if key in summary:
                    keyStats_df = keyStats_df.append(get_single_row(key, summary, symbol), sort=False)
                key = 'financialData'
                if key in summary:
                    finStats_df = finStats_df.append(get_single_row(key, summary, symbol), sort=False)

                key = 'earningsTrend'
                if key in summary:
                    eps_est, rev_est, eps_trend, eps_rev = parse_earnings_trend(summary, symbol)
                    earningsEst_df = earningsEst_df.append(eps_est, sort=False)
                    revenueEst_df = revenueEst_df.append(rev_est, sort=False)
                    epsTrend_df = epsTrend_df.append(eps_trend, sort=False)
                    epsRevisions_df = epsRevisions_df.append(eps_rev, sort=False)

                key = 'netSharePurchaseActivity'
                netSharePA_df = netSharePA_df.append(get_single_row(key, summary, symbol), sort=False)

                key = 'majorHoldersBreakdown'
                if key in summary:
                    majorHolders_df = majorHolders_df.append(
                        get_single_row(key, summary, symbol), sort=False)

                root, sub = 'institutionOwnership', 'ownershipList'
                if root in summary and sub in summary[root] and len(summary[root][sub]):
                    ownershipList_df = ownershipList_df.append(
                        create_normalized_df(summary[root][sub], symbol), sort=False)

                root, sub = 'fundOwnership', 'ownershipList'
                if root in summary and sub in summary[root] and len(summary[root][sub]):
                    fundOwnership_df = fundOwnership_df.append(
                        create_normalized_df(summary[root][sub], symbol), sort=False)

                root, sub = 'upgradeDowngradeHistory', 'history'
                if root in summary and sub in summary[root] and len(summary[root][sub]):
                    recommendHistory_df = recommendHistory_df.append(
                        create_normalized_df(summary[root][sub], symbol), sort=False)

                root, sub = 'insiderHolders', 'holders'
                if root in summary and sub in summary[root] and len(summary[root][sub]):
                    insiderHolders_df = insiderHolders_df.append(
                        create_normalized_df(summary[root][sub], symbol), sort=False)

                root, sub = 'recommendationTrend', 'trend'
                if root in summary and sub in summary[root] and len(summary[root][sub]):
                    recommendTrend_df = recommendTrend_df.append(
                        create_normalized_df(summary[root][sub], symbol), sort=False)

                root, sub = 'earningsHistory', 'history'
                if root in summary and sub in summary[root] and len(summary[root][sub]):
                    earningsHistory_df = earningsHistory_df.append(
                        create_normalized_df(summary[root][sub], symbol), sort=False)

                root, sub = 'indexTrend', 'estimates'
                if root in summary and sub in summary[root] and len(summary[root][sub]):
                    route = summary[root][sub]
                    df = clean_up_fmt(json_normalize(route))
                    df = df.T.rename(columns=df.period).drop(['period'])
                    df['peRatio'] = summary['indexTrend']['peRatio']['raw']
                    df['pegRatio'] = summary['indexTrend']['pegRatio']['raw']
                    df['symbol'] = summary['indexTrend']['symbol']
                    indexTrend_df = df

            print('{} Full unpack for {}'.format(i, symbol))
            i += 1

        # overriden daily
        csv_store(set_storeDate(profile_df, d), 'summary-categories/', csv_ext.format('assetProfile'))
        csv_store(set_storeDate(officers_df, d), 'summary-categories/', csv_ext.format('companyOfficers'))
        # financials -> need to find a way to append to this file
        csv_store(set_storeDate(finStmtBS_df, d), 'summary-categories/', csv_ext.format('financials-BS'))
        csv_store(set_storeDate(finStmtIS_df, d), 'summary-categories/', csv_ext.format('financials-IS'))
        csv_store(set_storeDate(finStmtCF_df, d), 'summary-categories/', csv_ext.format('financials-CF'))
        # new additions
        csv_store(set_storeDate(majorHolders_df, d), 'summary-categories/', csv_ext.format('majorHoldersBreakdown'))
        csv_store(set_storeDate(ownershipList_df, d), 'summary-categories/', csv_ext.format('institutionOwnership'))
        csv_store(set_storeDate(fundOwnership_df, d), 'summary-categories/', csv_ext.format('fundOwnership'))
        csv_store(set_storeDate(recommendHistory_df, d), 'summary-categories/', csv_ext.format('upgradeDowngradeHistory'))
        csv_store(set_storeDate(insiderHolders_df, d), 'summary-categories/', csv_ext.format('insiderHolders'))
        csv_store(set_storeDate(earningsHistory_df, d), 'summary-categories/', csv_ext.format('earningsHistory'))

        # saved a record per day
        fname = csv_ext.format(str(d))
        csv_store(set_storeDate(keyStats_df, d), 'summary-categories/defaultKeyStatistics/', fname)
        csv_store(set_storeDate(finStats_df, d), 'summary-categories/financialData/', fname)
        csv_store(set_storeDate(earningsEst_df, d), 'summary-categories/earningsEstimate/', fname)
        csv_store(set_storeDate(revenueEst_df, d), 'summary-categories/revenueEstimate/', fname)
        csv_store(set_storeDate(epsTrend_df, d), 'summary-categories/epsTrend/', fname)
        csv_store(set_storeDate(epsRevisions_df, d), 'summary-categories/epsRevisions/', fname)
        csv_store(set_storeDate(netSharePA_df, d), 'summary-categories/netSharePurchaseActivity/', fname)
        csv_store(set_storeDate(indexTrend_df, d), 'summary-categories/indexTrend/', fname)
        csv_store(set_storeDate(recommendTrend_df, d), 'summary-categories/recommendationTrend/', fname)
