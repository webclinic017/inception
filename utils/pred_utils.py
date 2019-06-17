import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from utils.TechnicalDS import TechnicalDS

pd.options.display.float_format = '{:,.3f}'.format
mpl.rcParams['figure.figsize'] = [5.0, 3.0]
mpl.rcParams['font.size'] = 8
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'large'

def get_top_predictions(pred_df, as_of_date, pred_classes, min_confidence):
    """ return top recommendatins by label as of a given date """
    label_mask = (pred_df.pred_class.isin(pred_classes)) & (pred_df.confidence > min_confidence)
    idx = pred_df.index.unique()[as_of_date]
    sort = True if np.mean(pred_classes) < 2.5 else False
    top_pred = pred_df.loc[(pred_df.index == idx) & label_mask].sort_values(by=['pred_label', 'confidence'], ascending=sort)
    
    return top_pred


def get_most_frequent_preds(pred_df, study_dates, top_pred, pred_classes, treshold=0.6):
    """ return most frequent predictions of a given class for a study period """
    # print(f'Most frequent predictions as of {study_dates[-1]} for classes {pred_classes}')
    last_xdays_pred = pred_df.loc[study_dates]
    last_xdays_pred = last_xdays_pred.loc[
        last_xdays_pred.symbol.isin(list(top_pred.symbol)), 
        ['symbol', 'pred_class', 'confidence']].reset_index()
    most_freq_df = last_xdays_pred.groupby(
        by=['symbol', 'pred_class']).agg(['count', 'mean']).reset_index()
    filter_mask = most_freq_df['pred_class'].isin(pred_classes) &\
    (most_freq_df[('confidence', 'count')] > int(len(study_dates) * treshold))
    result = most_freq_df.loc[filter_mask].sort_values(
        by=[('confidence', 'count'), ('confidence', 'mean')], 
        ascending=False)

    return result


def get_study_date_range(pred_df, as_of_date, study_period):
    """ 
    return date range for a study period, as of = prediction, 
    study period = number of days to observe stability of predictions    
    """
    if as_of_date == -1:
        return pred_df.index.unique()[study_period + as_of_date:]
    else:
        return pred_df.index.unique()[(study_period + as_of_date + 1):as_of_date+1]


def stop_loss(df, long, max_loss):
    truth_df = (df < 1 - max_loss) if long else (df > 1 + max_loss)
    pos = truth_df[truth_df == True]
    if len(pos): df.loc[pos.index[0]:] = df.loc[pos.index[0]]
    return df


def plot_symbol_hist_pred(pred_symbol, clean_px, tail, context, pred_df, labels):

    fig, axes = plt.subplots(nrows=3, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.5)

    hist_px = clean_px[pred_symbol].copy().dropna()
    hist_px.name = 'close'
    look_ahead = context['look_ahead']
    pct_chg_df = hist_px.pct_change(look_ahead).tail(tail)
    pct_chg_df.name = 'pct_chg'
    # hist_px = pd.concat([hist_px, pct_chg_df], axis=1, sort=False)
    co_pred = pred_df.loc[(pred_df.symbol == pred_symbol) & (pred_df.index.isin(pct_chg_df.index)), ['pred_class'] + labels]
    # hist_pred = pd.concat([co_pred['pred_class'], pct_chg_df], axis=1, sort=False)
    
    # forward looking returns
    hist_px.tail(tail).plot(
        title=f'{pred_symbol}: Historical share  price', 
        grid=True, rot=0,
        ax=axes[0], 
    )

    # forward looking returns
    pct_chg_df.tail(tail).plot.area(
        title=f'{int(np.mean(context["look_ahead"]))} day historical return',  
        grid=True, rot=0, color='orange', stacked=False,
        ax=axes[1],
    )

    # probability distribution
    co_pred[labels].plot.area(
        title=f'{int(np.mean(context["look_ahead"]))} day forward looking probability distribution', 
        ylim=(0, 1), cmap='RdYlGn', rot=0, 
        ax=axes[2],
    )
    # axes[2].legend(**plt_dict)
    
def get_dispersion_stats(key, feats_dict, sel_df, symbols):
    df = feats_dict[key]
    tgt_df = df.loc[sel_df.index, symbols]
    res_df = tgt_df[~sel_df.isna()].describe().T
#     print(res_df.sort_values(by='mean'))
    desc_df = res_df.describe().loc[['25%','50%','75%'], '50%']
    desc_df.name = key
    return desc_df