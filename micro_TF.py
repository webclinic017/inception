# imports
import time, os, sys
from tqdm import tqdm

# from matplotlib import pyplot as plt
from utils.basic_utils import *
from utils.basic_utils import *
from utils.fundamental import chain_outlier, get_focus_tickers
from utils.pricing import load_px_close, get_return_intervals
from utils.pricing import dummy_col, discret_rets, sample_wgts
from utils.pricing import px_mom_feats, px_mom_co_feats_light
from utils.pricing import eq_wgt_indices, to_index_form, rename_col

import time, os, sys
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.metrics import precision_score, roc_auc_score

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, Adamax, Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

# environment variables
bench = '^GSPC'
y_col = 'fwdReturn'
tickers = excl(config['companies'], [])

context = {
    'ml_path': './ML/',
    'model_name': 'micro_TF.h5',
    'tmp_path': './tmp/',
    'ds_name': 'co-technicals-ds',
    'px_close': 'universe-px-ds',
    'trained_cols': 'micro_TF_train_cols.npy',
    'look_ahead': 120,
    'look_back': 252,
    'smooth_window': 10,
    'load_ds': True,
    'scale': True,
    'test_size': .05,
    'verbose': True,
    's3_path': 'recommend/micro_ML/',
    'neuron_mult': 30,
    'max_iter': 400,
    'l2_reg': 1e-2,
}

px_close = load_px_close(
    context['tmp_path'], context['px_close'], context['load_ds']).drop_duplicates()
print('px_close.info()', px_close.info())

prices = px_close.dropna(subset=[bench])[tickers]
look_ahead = context['look_ahead']
cut_range = get_return_intervals(prices, look_ahead, tresholds=[0.25, 0.75])
fwd_ret_labels = ["bear", "short", "neutral", "long", "bull"]
print(f'Return intervals {cut_range}')

# latest quotes, profile, and industries
dates = read_dates('quote')
tgt_date = dates[-1] # last date saved in S3
print(f'Target date: {tgt_date}')

quotes = load_csvs('quote_consol', [tgt_date])
quotes.set_index('symbol', drop=False, inplace=True)

profile = load_csvs('summary_detail', ['assetProfile'])
profile.set_index('symbol', drop=False, inplace=True)


# MODEL SPECIIFIC FUNCTIONS
def create_pre_process_ds(context):

    tickers = context['tickers']
    sectors = profile.loc[profile.symbol.isin(tickers)].sector.unique()
    industries = profile.loc[profile.symbol.isin(tickers)].industry.unique()
    print(f'Sectors: {sectors.shape[0]}, Industries: {industries.shape[0]}')

    indices_df = pd.concat(
        [eq_wgt_indices(profile, px_close, 'sector', sectors, subset=tickers),
        eq_wgt_indices(profile, px_close, 'industry', industries, subset=tickers),
        to_index_form(px_close[bench], bench)],
        axis=1).drop_duplicates()

    # create price momentum features
    tmp_path = context['tmp_path']
    ds_name = context['ds_name']

    super_list = []
    for i, ticker in tqdm(enumerate(tickers)):
        try:
            close = px_close[ticker].dropna()
            ft_df = px_mom_feats(close, ticker, incl_name=False)
            if ticker in profile.symbol.unique():
                top_groups = tuple([bench, profile.loc[ticker, 'sector']])
                co = px_mom_co_feats_light(close, indices_df, top_groups)
                ft_df = pd.concat([ft_df, co.loc[ft_df.index, :]], axis=1)
                super_list.append(ft_df.copy())
            else: print(ticker, 'missing profile, skipping')
        except Exception as e:
            print("Exception: {0} {1}".format(ticker, e))

    joined_df = pd.concat(super_list, axis=0)
    joined_df = chain_outlier(joined_df, None)

    # basic impute and scaling
    scale_on = context['scale']
    scaler = StandardScaler()
    num_cols = numeric_cols(joined_df)
    joined_df.loc[:, num_cols] = joined_df[num_cols].replace([np.inf, -np.inf, np.nan], 0)
    if scale_on: joined_df.loc[:, num_cols] = scaler.fit_transform(joined_df[num_cols])

    # add categoricals
    joined_df = dummy_col(joined_df, 'sector', shorten=True)

    return joined_df

def train_ds(context):

    verbose = context['verbose']
    ml_path, model_name = context['ml_path'], context['model_name']
    trained_cols = context['trained_cols']
    test_size = context['test_size']
    look_ahead, look_back, smooth_window = context['look_ahead'], context['look_back'], context['smooth_window']
    f'{look_ahead} days, {look_back} days, {smooth_window} days'

    joined_df = create_pre_process_ds(context)

    # if we want to limit training set
    # index = joined_df.sort_index().index.unique()[-look_back:]
    # joined_df = joined_df.loc[index, :]
    # joined_df.shape

    # calculation of forward returns
    Y = px_close.loc[:, tickers].pct_change(look_ahead).shift(-look_ahead)
    Y = Y.rolling(smooth_window).mean() # smooth by the same length
    Y = Y[~(Y.isna().all(1))]
    Y = Y.loc[joined_df.index.unique(), :]

    # reshapes to include symbol in index in additional to date
    Y_df = Y.loc[joined_df.index.unique().sortlevel()[0], tickers]
    Y_df = Y_df.stack().to_frame().rename(columns={0: y_col})
    # somwhat repetitive with steps above but performs faster
    Y_df.index.set_names(['storeDate', 'symbol'], inplace=True)
    print('Y_df.shape', Y_df.shape)

    # re-index processed df on storeDate and symbol to have similar indices
    joined_df.index.set_names('storeDate', inplace=True)
    joined_df.set_index(['symbol'], append=True, inplace=True)
    print('joined_df.shape', joined_df.shape)

    # add Y values to processed df fast without having to loop
    joined_df.loc[:, y_col] = Y_df.loc[joined_df.index, y_col]

    # joined_df.loc[(slice(None), 'AAPL'), y_col].plot() # visualize smoothing
    # joined_df.groupby('symbol')[y_col].mean().sort_values() # rank long-term mean performance

    # discretize Y-variable
    joined_df.dropna(subset=[y_col], inplace=True)
    joined_df[y_col] = discret_rets(joined_df[y_col], cut_range, fwd_ret_labels)
    print('joined_df.shape', joined_df.shape)
    print(sample_wgts(joined_df[y_col]))

    joined_df.dropna(subset=[y_col], inplace=True)
    joined_df.loc[:, y_col] = joined_df[y_col].astype(str)

    days = len(joined_df.index.levels[0].unique())
    print(f'Training for {days} dates, {round(days/252, 1)} years')

    # joined_df.loc[(slice(None), 'TAL'), y_col].value_counts() # look at a specific security distribution
    train_df = joined_df.reset_index(drop=True)
    train_df.shape

    # create training and test sets
    X, y = train_df.drop(columns=y_col), train_df[y_col]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        break # just one split

    # skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    # for train_index, test_index in skf.split(X, y):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #     break

    # Keras Model
    neuron_mult = context['neuron_mult']
    max_iter = context['max_iter']
    l2_reg = context['l2_reg']
    # units = X_train.shape[1] * neuron_mult
    units = 500
    print(f'max_iter: {max_iter}, l2_reg: {l2_reg}, units: {units}')

    y_train_oh = pd.get_dummies(y_train)[fwd_ret_labels]
    y_test_oh = pd.get_dummies(y_test)[fwd_ret_labels]

    # keras.regularizers.l2()

    model = Sequential()
    model.add(Dense(units, activation='tanh', input_dim=X_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(Dense(units, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(units, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(units, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(units, activation='tanh'))
    model.add(Dropout(0.1))    
    model.add(Dense(len(pd.unique(y_train)), activation='softmax'))

    opt = Adam()
    # opt = Adagrad() #lr adapted relative to how frequently a parameter gets updated, the more updates the smaller the lr
    # opt = Adadelta() #more robust extension of Adagrad, adapts lr based on a moving window of gradient updates, instead of accumulating all past gradients
    # opt = Adamax() #variant of Adam based on the infinity norm
    # opt = Nadam() #essentially RMSprop with momentum, Nadam is Adam RMSprop with Nesterov momentum
    # opt = RMSprop() #optimizer is usually a good choice for recurrent neural networks

    es = EarlyStopping(
        monitor='loss', patience=10, restore_best_weights=True, verbose=1)

    ml_path, model_name = context['ml_path'], context['model_name']
    fname = ml_path + model_name
    checkpointer = ModelCheckpoint(filepath=fname, verbose=1, save_best_only=True)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(
        X_train, y_train_oh, validation_data=(X_test, y_test_oh),
        epochs=max_iter,
        batch_size=200,
        callbacks=[
                  es,
                  checkpointer,
              ])
    score = model.evaluate(X_test, y_test_oh)
    print(f'Test loss: {score[0]}, Test accuracy: {score[1]}')

    # save training columns
    np.save(ml_path + trained_cols, X_train.columns) # save feature order
    print(f'X_train.shape {X_train.shape}, columns: {list(X_train.columns)}')
    print('Saved: ', ml_path + trained_cols)

    # save model to drive
    model.save(fname)
    print('Saved ', fname)


def predict_ds(context):

    ml_path = context['ml_path']
    model_name = context['model_name']
    trained_cols = context['trained_cols']

    joined_df = create_pre_process_ds(context)
    pred_X = joined_df.loc[joined_df.sort_index().index[-1], :]
    print('pred_X.shape', pred_X.shape)

    # ensure prediction dataset is consistent with trained model
    train_cols = np.load(ml_path + trained_cols) # save feature order
    missing_cols = [x for x in train_cols if x not in pred_X.columns]
    if len(missing_cols):
        print(f'Warning missing columns: {missing_cols}')
        pred_X = pd.concat([pred_X, pd.DataFrame(columns=missing_cols)], axis=1)
        pred_X[missing_cols] = 0

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
    csv_store(s3_df, s3_path, csv_ext.format(tgt_date))

    return pred_df


if __name__ == '__main__':
    hook = sys.argv[1]

    # Smaller subset for testing
    tgt_sectors = [
        # 'Technology',
        'Healthcare',
    #     'Industrials',
    #     'Basic Materials',
        # 'Consumer Cyclical',
    #     'Financial Services',
        # 'Consumer Defensive',
    #     'Real Estate',
    #     'Utilities',
        # 'Communication Services',
    #     'Energy',
    ]

    size_df = get_focus_tickers(quotes, profile, tgt_sectors)
    # ind_count = size_df.groupby('industry').count()['marketCap']
    # tgt_industries = list(ind_count.loc[ind_count > ind_count.median() - 1].index)
    # tickers = list(profile.loc[profile.industry.isin(tgt_industries), 'symbol'])
    tickers = list(profile.loc[profile.sector.isin(tgt_sectors), 'symbol'])
    # tickers = list(quotes.loc[quotes.quoteType == 'EQUITY', 'symbol'])
    # tickers = list(size_df.index)

    context['tickers'] = tickers
    print(f'{len(context["tickers"])} companies')

    if hook == 'train':
        # train with 50 random tickers, keep model small, same results
        print('Training...')
        train_ds(context)

    elif hook == 'predict':
        print('Predicting...')
        predict_ds(context)

    else: print('Invalid option, please try: train or predict')
