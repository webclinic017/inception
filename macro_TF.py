# %%
import sys
import os
from utils.basic_utils import *
from utils.pricing import *
from utils.fundamental import *
from utils.MacroDS import MacroDS
from utils.BaseDS import BaseDS

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.layers import BatchNormalization

from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# %%
def pre_process_ds(context):
    temp_path = context['tmp_path']
    px_vol_fname = context['px_vol_ds']
    benchSL, sectorSL, riskSL, rateSL, bondSL, commSL, currSL = (
        macro_ds.universe_dict[x]
        for x in (
            'benchmarks', 'sectors', 'risk', 'rates',
            'bonds', 'commodities', 'currencies')
    )

    keep_bench = excl(benchSL, ['^STOXX50E', '^AXJO'])
    keep_fx = excl(currSL, ['HKD=X', 'MXN=X', 'AUDUSD=X', 'NZDUSD=X', 'TWD=X', 'CLP=X', 'KRW=X'])
    keep_sect = excl(sectorSL,['SPY', 'QQQ', 'DIA', 'IWM', 'XLC', 'XLRE'])
    keep_bonds = ['LQD', 'HYG']
    include = riskSL + keep_bench + keep_sect + rateSL + keep_fx + keep_bonds
    raw_df = macro_ds.stitch_instruments(symbols=include, name=True, axis=1)

    # clean up empty columns
    empty_cols = raw_df.iloc[-1].loc[raw_df.iloc[-1].isna()].index
    remove_cols = excl(empty_cols, [y_col])
    raw_df.drop(columns=remove_cols, inplace=True)
    print('Pre-clean up shape:', raw_df.shape)

    slim_cols = raw_df.isna().where(raw_df.isna() == True)\
        .count().sort_values(ascending=False)
    # keep columns with less than 400 NAs
    keep_cols = list(slim_cols.loc[slim_cols < 400].index)
    raw_df = raw_df[keep_cols]
    print('Post-clean up shape:', raw_df.shape)
    fill_on = context['fill']
    scaler_on = context['scale']

    scaler = StandardScaler()
    X_cols = excl(raw_df.columns, [y_col])
    raw_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if scaler_on:
        raw_df[X_cols] = scaler.fit_transform(raw_df[X_cols])
    if fill_on:
        raw_df.loc[:, X_cols] = raw_df.fillna(method=fill_on)

    return raw_df


# %%
def get_train_test_sets(context):

    raw_df = pre_process_ds(context)
    # discretize forward returns into classes
    cut_range = macro_ds.return_intervals(tresholds=[0.60, 0.90])
    BaseDS.labelize_ycol(raw_df, y_col, cut_range, labels)

    X_cols = excl(raw_df.columns, [y_col])

    imputer_on = context['impute']
    imputer = SimpleImputer(
        missing_values=np.nan,
        strategy='median', copy=False)
    if imputer_on:
        raw_df.loc[:, X_cols] = imputer.fit_transform(raw_df[X_cols])
    else:
        raw_df.dropna(inplace=True)

    test_sz = context['test_size']
    # create training and test sets
    X, y = raw_df.drop(columns=y_col), raw_df[y_col]
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_sz, random_state=42)
    # just one split
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        break

    return X_train, X_test, y_train, y_test


# %%
def train_ds(context):

    max_iter = context['max_iter']
    l2_reg = context['l2_reg']
    units = context['units']
    trained_cols = context['trained_cols']

    X_train, X_test, y_train, y_test = get_train_test_sets(context)
    y_train_oh = pd.get_dummies(y_train)[labels]
    y_test_oh = pd.get_dummies(y_test)[labels]

    # Keras Model
    y_train_oh = pd.get_dummies(y_train)[labels]
    y_test_oh = pd.get_dummies(y_test)[labels]

    model = Sequential()
    model.add(Dense(units, input_dim=X_train.shape[1], use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(int(units/2), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(pd.unique(y_train)), activation='softmax'))
    # keras.regularizers.l2(l2_reg)

    ml_path, model_name = context['ml_path'], context['model_name']
    fname = ml_path + model_name

    # save training columns
    # save feature order
    np.save(ml_path + trained_cols, X_train.columns)
    print(f'X_train.shape {X_train.shape}, columns: {list(X_train.columns)}')
    print('Saved: ', ml_path + trained_cols)

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)
    csv_logger = CSVLogger(f'{ml_path}macro-train-{macro_ds.tgt_date}.log')

    opt = Adam()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt, metrics=['accuracy'])

    history = model.fit(
        X_train, y_train_oh, validation_data=(X_test, y_test_oh),
        epochs=max_iter,
        batch_size=64,
        callbacks=[es, checkpointer, csv_logger])

    score = model.evaluate(X_test, y_test_oh)
    print(f'Loss: {score[0]}, Accuracy: {score[1]}')


# %%
def predict_ds(context):

    ml_path = context['ml_path']
    model_name = context['model_name']
    trained_cols = context['trained_cols']
    look_back = context['look_back']

    joined_df = pre_process_ds(context)
    pred_X = joined_df.loc[
        joined_df.sort_index().index.unique()[-look_back:], :]
    print('pred_X.shape', pred_X.shape)

    # ensure prediction dataset is consistent with trained model
    # save feature order
    train_cols = np.load(ml_path + trained_cols, allow_pickle=True)
    missing_cols = [x for x in train_cols if x not in pred_X.columns]
    if len(missing_cols):
        print(f'Warning missing columns: {missing_cols}')
        for c in missing_cols:
            pred_X[c] = 0

    sorted_cols = list(train_cols)
    print('pred_X.shape', pred_X[sorted_cols].shape)
    pred_df = macro_ds.px_vol_df['close']\
        .loc[pred_X.index, macro_ds.bench].to_frame()

    # Load model
    fname = ml_path + model_name
    model = load_model(fname)
    print('Loaded', fname)
    # Predict
    preds = model.predict(pred_X[sorted_cols])
    preds_classes = model.predict_classes(pred_X[sorted_cols])
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
    csv_store(s3_df, s3_path, csv_ext.format(macro_ds.tgt_date))

    return pred_df


# %%
context = {
    'portion': 100e-2,
    'ml_path': './ML/',
    'model_name': 'macro_TF-21fwd.h5',
    'tmp_path': './tmp/',
    'px_vol_ds': 'universe-px-vol-ds.h5',
    'trained_cols': 'macro_TF_train_cols-21fwd.npy',
    'look_ahead': 21,
    'look_back': 120,
    'smooth_window': 1,
    'predict_batch': 252,
    'train_window': 250*3,
    'load_ds': True,
    'impute': True,
    'fill': 'ffill',
    'scale': True,
    'test_size': .10,
    's3_path': 'recommend/macro_ML/',
    'verbose': 2,
    'units': 500,
    'hidden_layers': 4,
    'max_iter': 100,
    'l2_reg': 1,
    'dropout': 0.5,
}

# %%
temp_path = context['tmp_path']
px_vol_fname = context['px_vol_ds']
macro_ds = MacroDS(
    path=temp_path,
    fname=px_vol_fname,
    load_ds=True,
    bench='^GSPC',
    look_ahead=context['look_ahead'],
    look_back=context['train_window'],
    invert_list=['EURUSD=X', 'GBPUSD=X'],
    include_list=['^VIX'],
    max_draw_on=True,
    )
y_col = f'{macro_ds.bench}{macro_ds.ycol_name}'
labels = macro_ds.forward_return_labels

# %%
if __name__ == '__main__':
    hook = sys.argv[1]
    if hook == 'train':
        print('Training...')
        train_ds(context)
    elif hook == 'predict':
        print('Predicting...')
        pred_df = predict_ds(context)
        print(pred_df.tail(10).round(3).T)
    else:
        print('Invalid option, please try: train or predict')


"""
# %%
# get latest pricing file from inferece instance
px_vol_ds = context['px_vol_ds']
tmp_path = context['tmp_path']
os.makedirs(tmp_path, exist_ok=True)
get_ipython().system('scp -i ~/.ssh/qc_infra.pem ubuntu@35.162.96.235:~/inception/tmp/{px_vol_ds} {tmp_path}{px_vol_ds}')

# %% Train
train_ds(context)

# %% Predict
pred_df = predict_ds(context)

# %% Read from S3
s3_path = context['s3_path']
pred_df = pd.read_csv(
    csv_load(f'{s3_path}{macro_ds.tgt_date}'),
    index_col='pred_date', parse_dates=True)

# %% Visualize
pred_df.columns
print('Prediction distribution')
pd.value_counts(pred_df.pred_label) / pd.value_counts(pred_df.pred_label).sum()

pre_class_cols = filter_cols(pred_df.columns, "pred_class")
pred_df.loc[:,[macro_ds.bench] + pre_class_cols].plot(
    secondary_y=pre_class_cols, figsize=(15, 5))
pred_df[labels].plot.area(
        title='Prediction Probabilities',
        figsize=(15, 2), ylim=(0, 1), cmap='RdYlGn')
f'Confidence Mean: {pred_df["confidence"].mean().round(3)}, Median {pred_df["confidence"].median().round(3)}'
"""