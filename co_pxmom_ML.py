# imports
from utils.basic_utils import *
from utils.pricing import *
from utils import ml_utils as mu

import time, os, sys
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.metrics import precision_score, roc_auc_score

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

# environment variables
bench = '^GSPC'
sec_windows, stds = [5, 20, 60], 1
pred_fwd_windows = [60]
inv = incl_px = incl_name = False
y_col = 'fwdReturn'
cuts = { '1d': [-1, -0.1, -.02, .02, .1, 1.] }
cut_range = cuts['1d']
fwd_ret_labels = ["bear", "short", "neutral", "long", "bull"]

print('Refreshing equity pricing...')
excl_list = [] # ['BHF', 'ERI']
symbols_list = excl(config['companies'], excl_list)
px_close = get_mults_pricing(symbols_list).drop_duplicates().dropna(subset=['AAPL'])
# save down to drive if refresh pricing
os.makedirs('tmp', exist_ok=True)
px_close.to_parquet('tmp/mult-co-px-ds')
print(px_close.info())

# latest quotes, profile, and industries
dates = read_dates('quote')
tgt_date = [dates[-1]] # last date saved in S3

quotes = load_csvs('quote_consol', tgt_date)
quotes.set_index('symbol', drop=False, inplace=True)

profile = load_csvs('summary_detail', ['assetProfile'])
profile.set_index('symbol', drop=False, inplace=True)

profile.drop(profile[profile.symbol.isin(excl_list)].index, inplace=True)

all_equities = quotes[quotes.quoteType == 'EQUITY'].symbol.unique()
print('Delta quote: ', set(symbols_list) - set(all_equities))
# reduced subset, if any
sub_equities = set(px_close.columns.tolist()).intersection(all_equities)
print('Delta reduced set: ', set(symbols_list) - set(sub_equities))

eqty_symbols = profile[profile.symbol.isin(sub_equities)].symbol.unique().tolist()
delta_symb = set(symbols_list) - set(eqty_symbols)
print('Delta profile: ', len(delta_symb), delta_symb)

# Create a frame of market, sector and industry index (once)
# for relative performance calculations
sel_profiles = profile[profile.symbol.isin(all_equities)]
sel_profiles.groupby(['sector', 'industry'])[['industry']].count()
sectors = sel_profiles.sector.unique()
industries = sel_profiles.industry.unique()

print(f'Sectors: {sectors.shape[0]}, Industries: {industries.shape[0]}')

indices_df = pd.concat([
    eq_wgt_indices(profile, px_close, 'sector', sectors, subset=eqty_symbols),
    eq_wgt_indices(profile, px_close, 'industry', industries, subset=eqty_symbols),
    to_index_form(get_symbol_pricing(bench)['close'], bench)
], axis=1).drop_duplicates()

# MODEL SPECIIFIC FUNCTIONS
def create_ds(context):
    print('create_ds')
    train_model = context['train_model']
    (path, ds_name) = context['ds_path_name']
    tickers = context['tickers']
    load_ds = context['load_ds']
    tail = 10**4 if train_model else 252*2

    if load_ds & os.path.isfile(path + '/' + ds_name):
        df_large = pd.read_parquet(path + '/' + ds_name)
        return df_large

    super_list = []
    for i, ticker in tqdm(enumerate(tickers)):
        try:
            close = px_close[ticker].dropna().tail(tail)
            ft_df = px_mom_feats(close, ticker, stds, inv, incl_px, sec_windows, incl_name)
            ft_df[y_col] = px_fwd_rets(close, ticker, pred_fwd_windows).mean(axis=1)

            df = get_symbol_pricing(ticker).tail(tail) #full retrieve
            top_groups = tuple([bench] + list(profile.loc[ticker, ['sector', 'industry']]))
            co = px_mom_co_feats(df, indices_df, top_groups)

            ft_df.loc[:, 'country'] = profile.loc[ticker,:].country
            ft_df.loc[:, 'currency'] = quotes.loc[ticker,:].currency
            ft_df = pd.concat([ft_df, co.loc[ft_df.index, :]], axis=1)
            super_list.append(ft_df)
            print('{} Adding {} to dataset'.format(i, ticker))
        except Exception as e:
            print("Exception: {0}\n{1}".format(ticker, e))
    df_large = pd.concat(super_list, axis=0)

    if train_model:
        os.makedirs(path, exist_ok=True)
        df_large.to_parquet(path + '/' + ds_name)
    print('df_large.shape {}'.format(df_large.shape))

    return df_large

def pre_process_ds(raw_df, context):
    print('pre_process_ds')
    train_model = context['train_model']
    fill_on, imputer_on, scaler_on = context['fill'], context['impute'], context['scale']
    categoricals, exclude = context['categoricals'], context['exclude']
    (path, train_cols) = context['trained_cols']
    test_sz, verbose = context['test_size'], context['verbose']

    # convert categorical columns
    for col in categoricals: raw_df = dummy_col(raw_df, col, shorten=True)
    raw_df.drop(columns=exclude[:-1], inplace=True) # remove all except symbol

    scaler = StandardScaler()
    imputer = SimpleImputer(
        missing_values=np.nan,
        strategy='median', copy=False)
    X_cols = excl(raw_df.columns, [exclude[-1] ,y_col]) #not needed

    raw_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = list(raw_df.select_dtypes('number').columns)
    if scaler_on: raw_df[X_cols] = scaler.fit_transform(raw_df[X_cols])

    pred_X = X_train = X_test = y_train = y_test = None
    if train_model:
        raw_df.drop(columns=exclude[-1], inplace=True) # remove symbol
        if fill_on: raw_df[X_cols].fillna(method=fill_on, inplace=True)

        # discretize forward returns into classes
        raw_df.dropna(subset=[y_col], inplace=True)
        raw_df.loc[:, y_col] = discret_rets(raw_df[y_col], cut_range, fwd_ret_labels)
        raw_df.dropna(subset=[y_col], inplace=True) # no nas in y_col
        print(sample_wgts(raw_df[y_col]))
        raw_df.loc[:, y_col] = raw_df[y_col].astype(str) # class as string

        if imputer_on: raw_df.loc[:, X_cols] = imputer.fit_transform(raw_df[X_cols])
        else: raw_df = raw_df.dropna()

        X, y = raw_df.drop(columns=y_col), raw_df[y_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_sz, random_state=42)
        np.save(path + train_cols, X_train.columns) # save feature order
    else:
        # feature for last date, pending to implement more flexibility
        pred_X = raw_df.loc[raw_df.index[-1], :].drop(columns=y_col).dropna(axis=0)

    [print(x.shape) for x in (pred_X, X_train, X_test, y_train, y_test) if x is not None]
    return pred_X, X_train, X_test, y_train, y_test

def train_ds(context):
    context['load_ds'] = True
    context['train_model'] = True
    grid_search = context['grid_search']
    verbose = context['verbose']
    (path, model_name) = context['ml_path']
    portion = context['portion']

    ds_df = create_ds(context)
    print(df.info(verbose=False))
    _, X_train, X_test, y_train, y_test = pre_process_ds(ds_df, context)

    features = X_train.shape[1]
    best_params = { # best from GridSearch
        'n_estimators': 25,
        'max_features': features,
        'max_depth': 30,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'random_state': 0,
        'n_jobs': -1}
    if grid_search:
        print('GridSearchCV for RandomForestClassifier')
        param_grid = {
            'n_estimators': [50],
            'max_features': ['sqrt', 'log2', features // 2, features // 3,],
            'max_depth': [30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 5, 10],
            'random_state': np.arange(0, 3, 1),}
        clf = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, n_jobs=-1,
                           cv=5, iid=True, verbose=verbose)
        clf.fit(X_train, y_train)
        if verbose:
            print_cv_results(
                clf, X_train, X_test, y_train, y_test, feat_imp=True, top=20)
        best_params = clf.best_params_
    clf1 = RandomForestClassifier(**best_params)
    clf1.fit(X_train, y_train)
    print('RandomForestClassifier scores: Train {}, Test {}'.format(
    clf1.score(X_train, y_train), clf1.score(X_test, y_test)))

    # ExtraTreesClassifier
    clf2 = ExtraTreesClassifier(
        n_estimators=50,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=5,
        n_jobs=-1)
    clf2.fit(X_train, y_train)
    print('ExtraTreesClassifier scores: Train {}, Test {}'.format(
    clf2.score(X_train, y_train), clf2.score(X_test, y_test)))

    for vote in ['hard', 'soft']:
        eclf = VotingClassifier(
            estimators=[('rf', clf1), ('et', clf2)],
            voting=vote)
        clf = eclf.fit(X_train, y_train)
        print('VotingClassifier scores Train {}, Test {}'.format(
                clf.score(X_train, y_train), clf.score(X_test, y_test)))
        os.makedirs(path, exist_ok=True)
        fname = path + model_name.format(vote)
        joblib.dump(clf, fname)
        print('Saved ', fname)

def predict_ds(context):
    context['load_ds'] = False
    context['train_model'] = False
    (path, model_name) = context['ml_path']
    verbose = context['verbose']
    (path, train_cols) = context['trained_cols']

    df_large = create_ds(context)
    pred_X, _, _, _, _ = pre_process_ds(df_large, context)
    print('predict_ds')
    print('pred_X.shape', pred_X.shape)

    # ensure prediction dataset is consistent with trained model
    trained_cols = np.load(path + train_cols) # save feature order
    missing_cols = [x for x in trained_cols if x not in pred_X.columns]
    pred_X = pd.concat([pred_X, pd.DataFrame(columns=missing_cols)], axis=1)
    pred_X[missing_cols] = 0
    pred_X = pred_X[list(trained_cols) + ['symbol']]

    pred_df = pd.DataFrame()
    pred_df['symbol'] = pred_X.symbol
    for vote in ['hard', 'soft']:
        fname = path + model_name.format(vote)
        clf = joblib.load(fname) # load latest models
        print('Loaded', fname)
        preds = clf.predict(pred_X.iloc[:, :-1])
        # preds = np.where(preds == 'nan', 'neutral', preds) #replace nan
        pred_class = np.array([fwd_ret_labels.index(x) for x in preds])
        pred_df[f'{vote}_pred_class'] = pred_class
        pred_df[f'{vote}_pred_label'] = preds
        if vote == 'soft':
            probs = clf.predict_proba(pred_X.iloc[:, :-1])
            pred_prob = np.argmax(probs, axis=1)
            pred_df[f'{vote}_confidence'] = [x[np.argmax(x)] for x in probs] # higest prob
            prob_df = pd.DataFrame(probs, index=pred_df.index, columns=clf.classes_)
            pred_df = pd.concat([pred_df, prob_df[fwd_ret_labels]], axis=1)

    # store in S3
    s3_path = context['s3_path']
    s3_df = pred_df.reset_index(drop=False)
    rename_col(s3_df, 'index', 'pred_date')
    csv_store(s3_df, s3_path, csv_ext.format(dates[-1]))

    return pred_df

# CONTEXT
context = {
    'tickers': eqty_symbols,
    'ml_path': ('./ML/', 'co_pxmom_ML_{}.pkl'),
    'ds_path_name': ('tmp', 'co-pxmom-large'),
    'trained_cols': ('./ML/', 'co_pxmom_train_cols.npy'),
    'load_ds': False,
    'portion': 100e-2,
    'categoricals': ['sector'],
    'exclude': ['industry', 'country', 'currency', 'symbol'],
    'fill': 'bfill',
    'impute': False,
    'scale': False,
    'test_size': .20,
    'grid_search': False,
    'verbose': 2,
    's3_path': 'recommend/co-pxmom/'
}

if __name__ == '__main__':
    hook = sys.argv[1]
    if hook == 'train':
        # train with 50 random tickers, keep model small, same results
        tickers = list(sample_sector_tickers(eqty_symbols, profile, sectors, 50).index)
        context['tickers'] = tickers
        print('Training model using:', context)
        train_ds(context)
    elif hook == 'predict':
        context['tickers'] = eqty_symbols
        print('Predicting model using:', context)
        pred_df = predict_ds(context)
    else: print('Invalid option, please try: train or predict')
