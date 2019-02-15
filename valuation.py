# imports
from basic_utils import *
import numpy as np

# lambdas
list_cols_excl = lambda allCols, exclCols: [x for x in allCols if x not in exclCols]
print_labels = lambda stmt, df: print("{} labels:\n{}".format(stmt, df.columns.tolist()))
valid_cols = lambda df, x: df.columns[df.columns.isin(x)].tolist()
get_FX = lambda fexs, tgt_fx: 1 / fxes_df[fxes_df.currency == tgt_fx].iloc[0].regularMarketPrice

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
    if len(dfA) > 0 and (dfQ.index[-1] > dfA.index[-1]):
        if not isBs:
            cols_to_add = list_cols_excl(dfQ.columns, exclCols)
            dfQ.loc[dfQ.index[-1], cols_to_add] = dfQ[cols_to_add].sum()
        return dfA.iloc[-3:].append(dfQ.iloc[-1]).sort_index()
    return dfA
