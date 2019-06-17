# imports

import sys
from utils.basic_utils import csv_store, csv_ext, numeric_cols
from utils.pricing import dummy_col
from utils.pricing import rename_col
from utils.fundamental import chain_outlier
from utils.TechnicalDS import TechnicalDS

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.layers import BatchNormalization

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# context
context = {
    'ml_path': './ML/',
    'tmp_path': './tmp/',
    'px_vol_ds': 'universe-px-vol-ds.h5',
    'model_name': 'micro_TF-all.h5',
    'trained_cols': 'micro_TF_train_cols-all.npy',
    'look_ahead': 120,
    'look_back': 120,
    'smooth': 1,
    'load_ds': True,
    'scale': True,
    'test_size': .05,
    'verbose': True,
    's3_path': 'recommend/micro_ML/',
    'units': 750, #850
    'max_iter': 100, #50
    'l2_reg': 0.01,
}

tech_ds = TechnicalDS(
    context['tmp_path'],
    context['px_vol_ds'],
    load_ds=True,
    tickers='All',
    look_ahead=context['look_ahead'],
    fwd_smooth=context['smooth'],
    max_draw_on=True)
y_col = tech_ds.ycol_name


def pre_process_ds(context):
    raw_df = tech_ds.stitch_companies_groups()
    print(f'Shape excluding NAs: {raw_df.shape}')
    symbols = raw_df.reset_index().set_index(['symbol']).index
    sector_map = tech_ds.profile.loc[tech_ds.tickers,'sector'].to_dict()
    raw_df.loc[:, 'sector'] = symbols.map(sector_map)
    raw_df = chain_outlier(raw_df, None)
    # basic impute and scaling
    scale_on = context['scale']
    scaler = StandardScaler()
    num_cols = numeric_cols(raw_df)
    if scale_on: raw_df.loc[:, num_cols] = scaler.fit_transform(
        raw_df[num_cols])
    # add categoricals
    raw_df.dropna(subset=['sector'], inplace=True)
    raw_df = dummy_col(raw_df, 'sector', shorten=True)
    return raw_df


def get_train_test_sets(context):

    test_size = context['test_size']
    joined_df = pre_process_ds(context)
    cut_range = tech_ds.return_intervals(tresholds=[0.4, 0.75])
    TechnicalDS.labelize_ycol(
        joined_df, tech_ds.ycol_name,
        cut_range, tech_ds.forward_return_labels)

    joined_df.dropna(inplace=True)
    days = len(joined_df.index.levels[0].unique())
    print(f'Training for {days} dates, {round(days/252, 1)} years')

    train_df = joined_df.reset_index(drop=True) # drop date
    print(f'Pre X, y columns: {train_df.columns}')

    # create training and test sets
    X, y = train_df.drop(columns=y_col), train_df[y_col]
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=42)
    # just one split
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        break

    return X_train, X_test, y_train, y_test


def train_ds(context):
    max_iter = context['max_iter']
    l2_reg = context['l2_reg']
    units = context['units']
    trained_cols = context['trained_cols']
    X_train, X_test, y_train, y_test = get_train_test_sets(context)
    y_train_oh = pd.get_dummies(y_train)[tech_ds.forward_return_labels]
    y_test_oh = pd.get_dummies(y_test)[tech_ds.forward_return_labels]

    # Keras Model
    model = Sequential()
    model.add(Dense(units, input_dim=X_train.shape[1], use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(units, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(units, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(int(units/2), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(len(pd.unique(y_train)), activation='softmax'))
    keras.regularizers.l2(l2_reg)

    opt = Adam()

    ml_path, model_name = context['ml_path'], context['model_name']
    fname = ml_path + model_name

    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
    checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
    csv_logger = CSVLogger('micro-train.log')

    # save training columns, feature order
    np.save(ml_path + trained_cols, X_train.columns)
    print(f'X_train.shape {X_train.shape}, columns: {list(X_train.columns)}')
    print('Saved: ', ml_path + trained_cols)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt, metrics=['accuracy'])
    history = model.fit(
        X_train, y_train_oh, validation_data=(X_test, y_test_oh),
        epochs=max_iter, batch_size=64,
        callbacks=[es, checkpointer, csv_logger])

    score = model.evaluate(X_test, y_test_oh)
    print(f'Test loss: {score[0]}, Test accuracy: {score[1]}')


def predict_ds(context):

    ml_path = context['ml_path']
    model_name = context['model_name']
    trained_cols = context['trained_cols']
    look_back = context['look_back']

    joined_df = pre_process_ds(context)
    joined_df.reset_index(level=1, inplace=True)
    pred_X = joined_df.loc[joined_df.sort_index().index.unique()[-look_back:], :]
    print('pred_X.shape', pred_X.shape)

    # ensure prediction dataset is consistent with trained model
    # save feature order
    train_cols = np.load(ml_path + trained_cols, allow_pickle=True)
    missing_cols = [x for x in train_cols if x not in pred_X.columns]
    if len(missing_cols):
        print(f'Warning missing columns: {missing_cols}')
        for c in missing_cols:
            pred_X[c] = 0

    sorted_cols = list(np.append(train_cols, ['symbol']))
    print('pred_X.shape', pred_X[sorted_cols].shape)

    pred_df = pd.DataFrame()
    pred_df['symbol'] = pred_X.symbol

    # Load model
    fname = ml_path + model_name
    model = load_model(fname)
    print('Loaded', fname)

    preds = model.predict(pred_X[sorted_cols].iloc[:, :-1])
    preds_classes = model.predict_classes(pred_X[sorted_cols].iloc[:, :-1])

    labels = tech_ds.forward_return_labels
    pred_df['pred_class'] = preds_classes
    pred_df['pred_label'] = list(map(lambda x: labels[x], preds_classes))
    probs = np.round(preds, 3)
    # higest prob
    pred_df['confidence'] = [x[np.argmax(x)] for x in probs]
    prob_df = pd.DataFrame(probs, index=pred_df.index, columns=labels)
    pred_df = pd.concat([pred_df, prob_df[labels]], axis=1)
    pred_df.index.name = 'pred_date'

    # store in S3
    s3_path = context['s3_path']
    s3_df = pred_df.reset_index(drop=False)
    rename_col(s3_df, 'index', 'pred_date')
    csv_store(s3_df, s3_path, csv_ext.format(tech_ds.tgt_date))

    return pred_df


if __name__ == '__main__':
    hook = sys.argv[1]

    if hook == 'train':
        print('Training...')
        train_ds(context)

    elif hook == 'predict':
        print('Predicting...')
        # price/share > 20 and vol > 300k shares
        quotes = tech_ds.quotes
        liquid_tickers = list(quotes.loc[
            (quotes.quoteType == 'EQUITY') &
            (quotes.regularMarketPrice > 20) &
            (quotes.averageDailyVolume3Month > 0.3e6)
            , 'symbol'])
        tech_ds.tickers = liquid_tickers
        predict_ds(context)

    else: print('Invalid option, please try: train or predict')
