import pandas as pd
import numpy as np

# utility functions
re_order_list = lambda a, b: [x for x in list(a) if x not in b]

def sample_sector_tickers(tickers, profile, sectors, n=100):
    show = ['sector']
    df = profile.loc[profile.symbol.isin(tickers)][show].copy()
    mapper = df.groupby(by=show).apply(lambda x: x.count() / df.count()).to_dict()['sector']
    df.loc[:, 'sect_weight'] = df.sector.map(mapper)
    while True:
        sample = df.sample(n, weights='sect_weight')
        if sample.sector.unique().shape[0] == sectors.shape[0]:
            return sample

def trim_df(df, portion):
    _, trim_df = train_test_split(df, test_size=portion, random_state=42)
    return trim_df

def print_cv_results(clf, test_train_sets, full_grid=False, feat_imp=True, top=20):
    (X_train, X_test, y_train, y_test) = test_train_sets
    cvres = clf.cv_results_
    print('BEST PARAMS:', clf.best_params_)
    print('SCORES:')
    print('clf.best_score_', clf.best_score_)
    print('train {}, test {}'.format(
        clf.score(X_train, y_train),
        clf.score(X_test, y_test)))
    if full_grid:
        print('GRID RESULTS:')
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(round(mean_score, 3), params)
    if feat_imp:
        feature_importances = clf.best_estimator_.feature_importances_
        print('SORTED FEATURES:')
        print(sorted(zip(
            feature_importances,
            list(X_train.columns)),
            reverse=True)[:top])
