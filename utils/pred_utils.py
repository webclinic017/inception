import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

pd.options.display.float_format = '{:,.3f}'.format
mpl.rcParams['figure.figsize'] = [5.0, 3.0]
mpl.rcParams['font.size'] = 8
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['figure.titlesize'] = 'large'

def get_top_predictions(pred_df, as_of_date, pred_classes, min_confidence):
    """ return top recommendatins by label as of a given date """
    label_mask = (pred_df.pred_class.isin(pred_classes)) & (pred_df.confidence > min_confidence)
    idx = pred_df.index.unique()[as_of_date]
    top_pred = pred_df.loc[(pred_df.index == idx) & label_mask].sort_values(by=['pred_label', 'confidence'], ascending=False)
    
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


def plot_symbol_hist_pred(pred_symbol, clean_px, context, pred_df, labels):

    fig, axes = plt.subplots(nrows=3, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    px_df = clean_px[pred_symbol]
    px_df.name = 'close'
    look_ahead = context['look_ahead']
    pct_chg_df = px_df.pct_change(look_ahead)
    pct_chg_df.name = 'pct_chg'

    co_pred = pred_df.loc[pred_df.symbol.isin([pred_symbol]), ['pred_class'] + labels]
    hist_pred = pd.concat([px_df.loc[pred_df.index.unique()], co_pred['pred_class']], axis=1, sort=False)
    
    # forward looking returns
    pct_chg_df.reindex(index=pred_df.index.unique()).plot(
        title=f'{pred_symbol} {int(np.mean(context["look_ahead"]))} day return', 
        grid=True, rot=0, ax=axes[0], 
        # figsize=(10, 2), 
    )

    # historical predictions
    hist_pred.dropna().plot(
        title=f'{pred_symbol} historical prediction',  
        secondary_y='pred_class', rot=0, ax=axes[1],
        # figsize=(10, 3),
    )

    # probability distribution
    co_pred[labels].plot.area(
        title=f'{pred_symbol} Prediction probabilities', 
        ylim=(0, 1), cmap='RdYlGn', rot=0, ax=axes[2],
        # figsize=(10, 2), 
    )