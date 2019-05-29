# imports

import sys, os
from utils.basic_utils import *
from utils.pricing import *
from utils.fundamental import *
from utils.BaseDS import BaseDS

pd.options.display.float_format = '{:,.2f}'.format

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from sklearn.ensemble import forest

from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.metrics import precision_score, roc_auc_score

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, Adamax, Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import CSVLogger

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# environment variables# environment variables
freq = '1d'
cuts = { '1d': [-1, -0.1, -.02, .02, .1, 1.] }
cut_range = cuts[freq]
fwd_ret_labels = ["bear", "short", "neutral", "long", "bull"]

# pricing, refresh once
benchSL, sectorSL, riskSL, rateSL, bondSL, commSL, currSL = \
    config['benchmarks'], config['sectors'], config['risk'], config['rates'], \
    config['bonds'], config['commodities'], config['currencies']
symbols_list = benchSL + sectorSL + riskSL + rateSL + bondSL + commSL + currSL

keep_bench = excl(benchSL, ['^STOXX50E', '^AXJO'])
keep_fx = excl(currSL, ['HKD=X', 'MXN=X', 'AUDUSD=X', 'NZDUSD=X', 'TWD=X', 'CLP=X', 'KRW=X'])
keep_sect = excl(sectorSL, ['SPY', 'QQQ', 'DIA', 'IWM', 'XLC', 'XLRE'])
keep_bonds = ['LQD', 'HYG']

include = riskSL + keep_bench + keep_sect + rateSL + keep_fx + keep_bonds
invert = ['EURUSD=X', 'GBPUSD=X']
incl_price = riskSL

bench = '^GSPC'
y_col = 'fwdReturn'
pred_fwd_windows = [20, 60, 120]
rate_windows = [20, 60]
sec_windows, stds = [5, 20, 60], 1

# MODEL SPECIFIC FUNCTIONS
def create_ds(px_close, context):
    print('create_ds')
    portion = context['portion']
    verbose = context['verbose']
    # average the return of the next periods
    # select only rows where Y variable is not null
    ds_idx = px_close.dropna(subset=[bench]).index

    df_large = pd.DataFrame()
    rate_ft_df = rate_feats(px_close[rateSL], rate_windows) # rate transforms
    df_large[rate_ft_df.columns] = rate_ft_df

    # price momentum transforms
    super_list = []
    for ticker in include:
        inv = ticker in invert
        incl_px = ticker in incl_price
        df = px_close[ticker]
        ft_df = px_mom_feats(df, ticker, stds, inv, incl_px, sec_windows)
        super_list.append(ft_df.drop_duplicates())
    df_large = pd.concat(super_list, axis=1).sort_index()
    df_large = df_large.loc[ds_idx, :] # drop NAs before discretizing

    return ds_idx, df_large

def pre_process_ds(raw_df, context):

    print('pre_process_ds')
    fill_on, scaler_on = context['fill'], context['scale']
    ml_path = context['ml_path']
    train_cols = context['trained_cols']
    test_sz, verbose = context['test_size'], context['verbose']

    scaler = StandardScaler()
    X_cols = excl(raw_df.columns, [y_col])

    raw_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if scaler_on: raw_df[X_cols] = scaler.fit_transform(raw_df[X_cols])
    if fill_on: raw_df.loc[:, X_cols] = raw_df.fillna(method=fill_on)

    return raw_df

def get_train_test_set(context):

    print(f'Benchmark: {bench}, Y: {y_col}, Include: {include}, invert: {invert}, include price: {incl_price}')

    imputer_on = context['impute']

    # create and pre-process datasets
    ds_idx, raw_df = create_ds(px_close, context)
    raw_df = pre_process_ds(raw_df, context)
    Y = px_fwd_rets(px_close.loc[ds_idx, bench], bench, pred_fwd_windows)
    raw_df[y_col] = Y

    # discretize forward returns into classes
    raw_df.dropna(subset=[y_col], inplace=True)
    raw_df.loc[:, y_col] = discret_rets(raw_df[y_col], cut_range, fwd_ret_labels)
    raw_df.dropna(subset=[y_col], inplace=True) # no nas in y_col
    print(sample_wgts(raw_df[y_col]))
    raw_df.loc[:, y_col] = raw_df[y_col].astype(str) # class as string

    X_cols = excl(raw_df.columns, [y_col])
    imputer = SimpleImputer(
        missing_values=np.nan,
        strategy='median', copy=False)
    if imputer_on: raw_df.loc[:, X_cols] = imputer.fit_transform(raw_df[X_cols])
    else: raw_df.dropna(inplace=True)

    return raw_df

def train_ds(context):

    test_sz = context['test_size']
    ml_path = context['ml_path']
    trained_cols = context['trained_cols']
    verbose = context['verbose']
    units = context['units']
    max_iter = context['max_iter']
    l2_reg = context['l2_reg']
    dropout = context['dropout']

    raw_df = get_train_test_set(context)

    X, y = raw_df.drop(columns=y_col), raw_df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=None)

    # save training columns
    np.save(ml_path + trained_cols, X_train.columns) # save feature order
    print(f'X_train.shape {X_train.shape}, columns: {list(X_train.columns)}')
    print('Saved: ', ml_path + trained_cols)

    # Keras Model
    y_train_oh = pd.get_dummies(y_train)[fwd_ret_labels]
    y_test_oh = pd.get_dummies(y_test)[fwd_ret_labels]

    keras.regularizers.l2(l=l2_reg)
    model = Sequential()
    model.add(Dense(units, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(int(units/2), activation='relu'))
    model.add(Dense(len(pd.unique(y_train)), activation='softmax'))

    ml_path, model_name = context['ml_path'], context['model_name']
    fname = ml_path + model_name

    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
    checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
    csv_logger = CSVLogger('macro-train.log')

    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model.fit(
    X_train, y_train_oh, validation_data=(X_test, y_test_oh),
    epochs=max_iter, batch_size=200,callbacks=[es, checkpointer, csv_logger])

    score = model.evaluate(X_test, y_test_oh)
    print(f'Loss: {score[0]}, Accuracy: {score[1]}')

    # save model to drive
    ml_path, model_name = context['ml_path'], context['model_name']
    fname = ml_path + model_name
    model.save(fname)
    print('Saved ', fname)

def predict_ds(context):
    ml_path = context['ml_path']
    model_name = context['model_name']
    trained_cols = context['trained_cols']
    pred_batch = context['predict_batch']

    ds_idx, raw_df = create_ds(px_close, context)
    pred_X = pre_process_ds(raw_df, context)
    pred_X = pred_X.iloc[-pred_batch:,:].dropna(axis=0)
    print('pred_X.shape', pred_X.shape)

    # ensure prediction dataset is consistent with trained model
    train_cols = np.load(ml_path + trained_cols, allow_pickle=True) # save feature order
    missing_cols = [x for x in train_cols if x not in pred_X.columns]
    if len(missing_cols):
        print(f'Warning missing columns: {missing_cols}')
        pred_X = pd.concat([pred_X, pd.DataFrame(columns=missing_cols)], axis=1)
        pred_X[missing_cols] = 0

    print('pred_X.shape', pred_X[train_cols].shape)

    pred_df = px_close.loc[pred_X.index, bench].to_frame()

    # Load model
    fname = ml_path + model_name
    model = load_model(fname)
    print('Loaded', fname)

    preds = model.predict(pred_X[train_cols])
    preds_classes = model.predict_classes(pred_X[train_cols])

    pred_df['pred_class'] = preds_classes
    pred_df['pred_label'] = list(map(lambda x: fwd_ret_labels[x], preds_classes))
    probs = np.round(preds,3)
    pred_prob = np.argmax(probs, axis=1)
    pred_df['confidence'] = [x[np.argmax(x)] for x in probs] # higest prob
    prob_df = pd.DataFrame(probs, index=pred_df.index, columns=fwd_ret_labels)
    pred_df = pd.concat([pred_df, prob_df[fwd_ret_labels]], axis=1)
    pred_df.index.name = 'pred_date'

    # store in S3
    s3_path = context['s3_path']
    s3_df = pred_df.reset_index(drop=False)
    rename_col(s3_df, 'index', 'pred_date')
    # tgt_date = str(pred_X.index[-1])
    csv_store(s3_df, s3_path, csv_ext.format(tgt_date))

    return pred_df

#context/config for training and prediction
context = {
    'portion': 100e-2,
    'ml_path': './ML/',
    'model_name': 'macro_TF.h5',
    'tmp_path': './tmp/',
    'px_vol_ds': 'universe-px-vol-ds.h5',
    'trained_cols': 'macro_TF_train_cols.npy',
    'look_ahead': 20,
    'look_back': 252,
    'smooth_window': 10,
    'predict_batch': 252,
    'load_ds': True,
    'impute': True,
    'fill': 'ffill',
    'scale': True,
    'test_size': .10,
    's3_path': 'recommend/macro_ML/',
    'verbose': 2,
    'units': 1000,
    'hidden_layers': 4,
    'max_iter': 400,
    'l2_reg': 0.5,
    'dropout': 0.5,
}

temp_path = context['tmp_path']
px_vol_fname = context['px_vol_ds']
base_ds = BaseDS(path=temp_path, fname=px_vol_fname, load_ds=True, )
# temporary workaround until load_px_close is @deprecated
px_close = base_ds.px_vol_df['close']

dates = read_dates('quote')
tgt_date = dates[-1] # last date saved in S3

if __name__ == '__main__':
    hook = sys.argv[1]
    if hook == 'train':
        print('Training...')
        train_ds(context)
    elif hook == 'predict':
        print('Predicting...')
        pred_df = predict_ds(context)
        print(pred_df.tail(5).round(3).T)
    else: print('Invalid option, please try: train or predict')
