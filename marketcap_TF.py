# %%
# imports
import time, os, sys
from tqdm import tqdm

# from matplotlib import pyplot as plt
from utils.basic_utils import *
from utils.pricing import load_px_close, discret_rets, sample_wgts
from utils.pricing import dummy_col, rename_col, px_fwd_rets, px_mom_feats, px_mom_co_feats_light
from utils.pricing import eq_wgt_indices, to_index_form, get_symbol_pricing, get_return_intervals
from utils.fundamental import pipe_transform_df, chain_divide, chain_scale
from utils.fundamental import chain_outlier, chain_post_drop, chain_wide_transform
from utils.fundamental import chain_share_multiple, chain_percent_total
from utils.fundamental import load_append_ds, get_daily_ts, numeric_cols, filter_cols
from utils.fundamental import get_focus_tickers
from utils.BaseDS import BaseDS


from sklearn import preprocessing

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.validation import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.metrics import precision_score, roc_auc_score

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import keras
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD, Adam, RMSprop
# from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

pd.options.display.float_format = '{:,.2f}'.format

#%%
# feature mapping for different datasets
ds_dict = load_config('./utils/fundamental_ds_dict.json')
# save_config(ds_dict, './utils/fund_ds_dict.json')

# pre-processing pipeline
fn_pipeline = {
    # 'fin_data': [chain_scale, chain_divide, chain_post_drop, chain_outlier],
    'fin_data': [chain_divide],
    # 'key_statistics': [chain_scale, chain_outlier],
    'key_statistics': [lambda x, y: x],
    # 'day_quote': [chain_divide, chain_scale, chain_outlier],
    'day_quote': [lambda x, y: x],
    'eps_trend': [chain_wide_transform, chain_share_multiple],
    'eps_estimates': [chain_wide_transform, chain_share_multiple],
    'rev_estimates': [chain_wide_transform],
    'eps_revisions': [chain_wide_transform],
    # 'spy_trend': [lambda x, y: x],
    # 'net_purchase':[lambda x, y: x],
    'rec_trend': [chain_wide_transform, chain_percent_total],
}

# %%
# environment variables
# latest quotes, profile, and industries
dates = read_dates('quote')
tgt_date = dates[-1] # last date saved in S3
print(f'Target date: {tgt_date}')

quotes = load_csvs('quote_consol', [tgt_date])
profile = load_csvs('summary_detail', ['assetProfile'])
quotes.set_index('symbol', drop=False, inplace=True)
profile.set_index('symbol', drop=False, inplace=True)

# save_config(context, './utils/marketcap_context.json')
context = load_config('./utils/marketcap_context.json')
context['fn_pipeline'] = fn_pipeline
bench = context['bench']
y_col = context['y_col']
temp_path = context['tmp_path']
px_vol_fname = context['px_vol_ds']

# PENDING: do the filtering here, outside of the functions
base_mask = (quotes.financialCurrency == 'USD') & (quotes.quoteType == 'EQUITY')
tickers = list(quotes.loc[base_mask].index)
# tickers = config['companies']
context['tickers'] = tickers

# %%
base_ds = BaseDS(path=temp_path, fname=px_vol_fname, load_ds=True, )
px_close = base_ds.px_vol_df['close']

stacked_px = px_close.stack().to_frame().rename(columns={0: 'close'}) # stack date + symbol
stacked_px.index.set_names(['storeDate', 'symbol'], inplace=True) # reindex
context['close_px'] = stacked_px

prices = px_close.dropna(subset=[bench])[tickers]
look_ahead = context['look_ahead']

x_scaler = PowerTransformer()
y_scaler = StandardScaler()

# %%
def load_files(context):
    """ load fundamental datasets from S3 or locally """
    # context variables
    key = context['key']
    load_ds = context['load_ds']
    load_dates = context['load_dates']
    tmp_path = context['tmp_path']

    # Load or append missing data to local dataset
    fname = tmp_path + key
    if load_ds & os.path.isfile(fname):
        daily_df = pd.read_parquet(fname)
        # compare and load missing dates
        missing_dates = list(
            set(daily_df.index.unique().date.astype(str))\
                .symmetric_difference(load_dates))
        if len(missing_dates) > 0: # retrieve missing dates
            append_df = get_daily_ts(key, ds_dict, missing_dates)
            daily_df = pd.concat([daily_df, append_df], axis=0) # append to daily
            # daily_df.drop_duplicates(inplace=True)
            daily_df.to_parquet(tmp_path + key) # and persist to drive for next time
    else:
        # file does not exist, retrieves all dates
        daily_df = get_daily_ts(key, ds_dict, load_dates)
        num_cols = excl(daily_df.columns, ['symbol', 'period'])
        daily_df.loc[:, num_cols] = daily_df[num_cols].astype(np.float32)
        # Make index a flat date, easier to index save down to drive if refresh pricing
        os.makedirs(tmp_path, exist_ok=True)
        # daily_df.drop_duplicates(inplace=True)
        daily_df.to_parquet(fname)
    
    daily_df.index.name = ds_dict[key]['index']
    daily_df.index = daily_df.index.date

    return daily_df

def create_ds(context):
    """ join and return all raw datasets """
    tickers = context['tickers']
    sectors = profile.loc[profile.symbol.isin(tickers)].sector.unique()
    industries = profile.loc[profile.symbol.isin(tickers)].industry.unique()
    print(f'Sectors: {sectors.shape[0]}, Industries: {industries.shape[0]}')
    # indices_df = pd.concat(
    #     [eq_wgt_indices(profile, px_close, 'sector', sectors, subset=tickers),
    #     eq_wgt_indices(profile, px_close, 'industry', industries, subset=tickers),
    #     to_index_form(px_close[bench], bench)],
    #     axis=1).drop_duplicates()

    s_l = []
    active_datasets = context['active_datasets']
    for key in active_datasets:
        print(f'adding {key}')
        context['key'] = key
        context['pre'] = key.split('_')[0] # append preffix
        context['ds_dict'] = ds_dict[key]
        load_dates = read_dates(ds_dict[key]['path'], '.csv')
        context['load_dates'] = load_dates
        df = load_files(context)
        df = df.loc[df.symbol.isin(tickers),:]
        df.index.name = 'storeDate'
        processed_df = pipe_transform_df(df, key, fn_pipeline, context)
        if 'symbol' in processed_df.columns:
            processed_df.set_index('symbol', append=True, inplace=True)
        df = processed_df.copy()
        s_l.append(df)
    dataset = pd.concat(s_l, axis=1)

    # add categoricals
    # joined_df = dummy_col(joined_df, 'sector', shorten=True)

    return dataset

def pre_process_ds(context):

    # cleans and scales datasets
    joined_df = create_ds(context)
    idx = pd.IndexSlice
    x_cols = context['x_cols']
    dataset = joined_df[x_cols]
    num_cols = numeric_cols(dataset)
    dataset.loc[:, num_cols] = dataset[num_cols].replace([np.inf, -np.inf, np.nan], 0)
    # dataset.describe().T
    base_mask = (quotes.financialCurrency == 'USD') & (quotes.quoteType == 'EQUITY')
    tickers = list(quotes.loc[base_mask].index)
    dataset = dataset.loc[idx[:, tickers], x_cols]
    dataset = dataset.loc[dataset[y_col] > 0]
    # dataset.describe().T
    X, y = dataset.drop(columns=y_col), dataset[y_col].values.reshape(-1,1)

    scaled_X = x_scaler.fit_transform(X)
    scaled_y = y_scaler.fit_transform(y)
    dataset[excl(x_cols, [y_col])] = scaled_X
    dataset[y_col] = scaled_y
    dataset.describe(include='all').T

    # drop the absolute minimum
    # dropna_cols = context['dropna_cols'] + [y_col]
    # dataset.dropna(subset=dropna_cols).isna().sum().sort_values()
    # dataset.dropna(subset=dropna_cols, inplace=True)
    # dataset.describe().T

    # then replace remaining NAs with 0, minimizes 0 bias
    # num_cols = numeric_cols(dataset)
    # dataset.replace([np.inf, -np.inf], 0, inplace=True)
    # dataset.fillna(0, inplace=True)
    # print(f'dataset.shape {dataset.shape}')

    # idx = pd.IndexSlice
    # dataset = raw_df[x_cols]
    # dataset = dataset.loc[dataset[y_col] > 0]
    # dataset = dataset.loc[idx[:, tickers], x_cols]
    # X, y = dataset.drop(columns=y_col), dataset[y_col].values.reshape(-1,1)

    # scale_on = context['scale']
    # if scale_on: 
    #     scaled_X = x_scaler.fit_transform(X)
    #     scaled_y = y_scaler.fit_transform(y)
    #     dataset[excl(x_cols, [y_col])] = scaled_X
    #     dataset[y_col] = scaled_y

    return dataset

def train_ds(context):

    dataset = pre_process_ds(context)
    test_size = context['test_size']
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_labels = train_dataset.pop(y_col)
    test_labels = test_dataset.pop(y_col)

    units = context['units']
    max_iter = context['epochs']
    # l2_reg = context['l2_reg']
    # dropout = context['dropout']
    trained_cols = context['trained_cols']
    ml_path, model_name = context['ml_path'], context['model_name']

    # save training columns
    np.save(ml_path + trained_cols, train_dataset.keys()) # save feature order
    print(f'X_train.shape {train_dataset.shape}, columns: {train_dataset.keys()}')
    print('Saved: ', ml_path + trained_cols)

    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(
        loss='mean_squared_error', optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error'])

    EPOCHS = 100
    ml_path, model_name = context['ml_path'], context['model_name']
    fname = ml_path + model_name

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10)
    checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=fname, verbose=1, save_best_only=True)

    history = model.fit(
        train_dataset, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=1,
        callbacks=[early_stop, checkpointer])

    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
    print(f'loss {loss}, mae {mae}, mse {mse}')

    # save model to drive
    model.save(fname)
    print('Saved ', fname)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    plot_history(history, y_col)

    test_predictions = model.predict(test_dataset).flatten()
    error = test_predictions - test_labels
    error = y_scaler.inverse_transform(error) / 10**9
    plot_scatter_error(test_labels, test_predictions, y_col)
    plot_error_hist(error, 50, (-100, 100), y_col)


def predict_ds(context):

    dataset = pre_process_ds(context)
    idx = pd.IndexSlice
    look_back = context['look_back']
    look_back_dates = dataset.index.levels[0][-look_back:]
    pred_X = dataset.loc[idx[look_back_dates, :], :]

    # ensure prediction dataset is consistent with trained model
    ml_path, model_name = context['ml_path'], context['model_name']
    pred_df = pd.DataFrame(index=pred_X.index)

    # Load model
    fname = ml_path + model_name
    model = keras.models.load_model(fname)
    print('Loaded', fname)

    # Predict
    actual = y_scaler.inverse_transform(pred_X[y_col].copy())
    predictions = model.predict(pred_X.drop(columns=[y_col])).flatten()
    unscaled = y_scaler.inverse_transform(predictions)

    # Update dataframe
    pred_df['predicted'] = unscaled
    pred_df['current'] = actual
    pred_df.index.name = ['pred_date', 'symbol']

    # store in S3
    s3_path = context['s3_path']
    s3_df = pred_df.reset_index(drop=False)
    rename_col(s3_df, 'index', 'pred_date')
    csv_store(s3_df, s3_path, csv_ext.format(tgt_date))

    return pred_df

def plot_scatter_error(true_vals, predicted_vals, y_col):
    plt.scatter(true_vals, predicted_vals)
    plt.xlabel(f'True Values [{y_col}]')
    plt.ylabel(f'Predictions [{y_col}]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])

def plot_error_hist(error, bins, x_range, y_col):
    plt.hist(error, bins)
    plt.xlim(x_range)
    plt.xlabel(f"Prediction Error [{y_col}]")
    _ = plt.ylabel("Count")

def plot_history(history, y_col):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel(f'Mean Abs Error [{y_col}]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
  """ plt.legend() """
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
  """ plt.ylim([0,20]) """
  plt.legend()
  plt.show()

# %%

if __name__ == '__main__':
    hook = sys.argv[1]

    if hook == 'train':
        print('Training...')
        train_ds(context)

    elif hook == 'predict':
        print('Predicting...')
        predict_ds(context)

    else: print('Invalid option, please try: train or predict')

# %% TEST TRAIN
# train_ds(context)

# %% TEST AND PLOT PREDICT
# pred_df = predict_ds(context)

# pred_df = pred_df / 10**9
# _curr, _pred = pred_df['current'].values, pred_df['predicted'].values
# error = _pred - _curr
# plot_scatter_error(_curr, _pred, y_col)
# plot_error_hist(error, 50, (-50, 50), y_col)

# %% OLD STABLE / QUICK AND DIRTY
# df = create_ds(context)
# joined_df = df.copy()
# joined_df.head()
# joined_df.describe().T
# idx = pd.IndexSlice
# x_cols = context['x_cols']
# dataset = joined_df[x_cols]
# num_cols = numeric_cols(dataset)
# dataset.loc[:, num_cols] = dataset[num_cols].replace([np.inf, -np.inf, np.nan], 0)
# dataset.describe().T
# base_mask = (quotes.financialCurrency == 'USD') & (quotes.quoteType == 'EQUITY')
# tickers = list(quotes.loc[base_mask].index)
# dataset = dataset.loc[idx[:, tickers], x_cols]
# dataset = dataset.loc[dataset[y_col] > 0]
# dataset.describe().T
# X, y = dataset.drop(columns=y_col), dataset[y_col].values.reshape(-1,1)

# x_scaler = PowerTransformer()
# y_scaler = StandardScaler()
# scaled_X = x_scaler.fit_transform(X)
# scaled_y = y_scaler.fit_transform(y)
# dataset[excl(x_cols, [y_col])] = scaled_X
# dataset[y_col] = scaled_y
# dataset.describe(include='all').T

# train_dataset = dataset.sample(frac=0.8,random_state=0)
# test_dataset = dataset.drop(train_dataset.index)
# len(train_dataset.keys())

# train_labels = train_dataset.pop(y_col)
# test_labels = test_dataset.pop(y_col)
# dataset.describe().T

# def build_model():
#   model = keras.Sequential([
#     layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
#     layers.Dense(64, activation=tf.nn.relu),
#     layers.Dense(1)
#   ])

#   optimizer = tf.keras.optimizers.RMSprop(0.001)

#   model.compile(loss='mean_squared_error',
#                 optimizer=optimizer,
#                 metrics=['mean_absolute_error', 'mean_squared_error'])
#   return model

# model = build_model()
# model.summary()

# %% TEST MODEL WORKS BEFORE TRAINING
# example_batch = train_dataset.iloc[:10]
# example_batch.T
# example_batch.describe().T
# example_result = model.predict(example_batch)
# example_result

#%%
# EPOCHS = 100
# ml_path, model_name = context['ml_path'], context['model_name']
# fname = ml_path + model_name

# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# checkpointer = keras.callbacks.ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)

# history = model.fit(
#   train_dataset, train_labels,
#   epochs=EPOCHS, validation_split=0.2, verbose=1,
#   callbacks=[early_stop, checkpointer])

# plot_history(history, y_col)
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()
# loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=0)
# test_predictions = model.predict(test_dataset).flatten()
# error = test_predictions - test_labels
# error = y_scaler.inverse_transform(error) / 10**9
# plot_scatter_error(test_labels, test_predictions, y_col)
# plot_error_hist(error, 50, (-100, 100), y_col)

#%% CHECK WETHER HISTORICAL PREDICTIONS TRACK ACTUAL VALUE
# symbol = 'AAPL'
# offset = -30
# idx = pd.IndexSlice
# sample_x = train_dataset.loc[idx[:,symbol], :].sort_index()
# print(sample_x.shape)
# sample_x.to_csv(f'{symbol}.csv')
# predictions = model.predict(sample_x).flatten()
# unscaled = y_scaler.inverse_transform(predictions) / 10**9
# actual_mktcap = joined_df.loc[sample_x.index, 'marketCap'] / 10**9
# plt.plot(unscaled)
# plt.plot(actual_mktcap.values)
# unscaled - actual_mktcap

#%% FILTER FOR PERIODS WITHOUT DATA
# joined_df.loc[idx['20190110':'20190120',symbol], :].iloc[-20:].T

# %% TAKES A LONG TIME WITH LARGE DATASETS
# sns.pairplot(dataset[dataset.columns[:5]], diag_kind="kde")