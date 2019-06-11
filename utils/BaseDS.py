import os
from tqdm import *
from utils.basic_utils import load_csvs, config, UNIVERSE
from utils.basic_utils import read_dates, numeric_cols, excl
from utils.pricing import get_symbol_pricing, shorten_name
import pandas as pd
import numpy as np


class BaseDS(object):

    universe_key = '^ALL'
    y_col_name = 'fwdRet'
    # forward_return_labels = ["bear", "short", "neutral", "long", "bull"]
    forward_return_labels = ["bear", "short", "negative", "positive", "long", "bull"]

    def __init__(
        self,
        path='../tmp/',
        fname='universe-px-vol-ds.h5',
        load_ds=True,
        bench='^GSPC',
        look_ahead=120,
        look_back=252*7,
        quantile=0.75
    ):

        self.path = path
        self.fname = fname
        self.load_ds = load_ds
        self.bench = bench
        self.look_ahead = look_ahead
        self.look_back = look_back
        self.quantile = quantile
        self.universe_dict = {k: config[k] for k in config['universe_list']}

        self.px_vol_df = self.load_px_vol_ds()
        self.clean_px = self.px_vol_df['close'].dropna(subset=[self.bench])

        # Quotes, profile, and industries for last date of dataset
        dates = self.clean_px.index.unique()
        self.tgt_date = dates[-1].strftime('%Y-%m-%d')
        print(f'Target date: {self.tgt_date}')

        quotes = load_csvs('quote', ['csv/' + self.tgt_date])
        keystats = load_csvs('summary_detail', ['defaultKeyStatistics/' + self.tgt_date])
        finstats = load_csvs('summary_detail', ['financialData/' + self.tgt_date])
        profile = load_csvs('summary_detail', ['assetProfile'])
        self.quotes = quotes.set_index('symbol', drop=False)
        self.keystats = keystats.set_index('symbol', drop=False)
        self.finstats = finstats.set_index('symbol', drop=False)
        self.profile = profile.set_index('symbol', drop=False)

    def load_px_vol_ds(self):
        """
        Refresh price and volume daily,
        used by most models with technical stats without refreshing
        """
        if os.path.isfile(self.path + self.fname) and self.load_ds:
            self.px_vol_ds = pd.read_hdf(self.path + self.fname, 'px_vol_df')
        else:
            # file does not exist, refreshes full dataset
            self.px_vol_ds = self.get_universe_px_vol(UNIVERSE)
            num_cols = numeric_cols(self.px_vol_ds)
            self.px_vol_ds.loc[:, num_cols] = self.px_vol_ds[num_cols].astype(np.float32)
            os.makedirs(self.path, exist_ok=True)
            self.px_vol_ds.to_hdf(self.path + self.fname, 'px_vol_df')
            # px_vol_ds.index = px_vol_ds.index.date
        print(self.px_vol_ds.info())
        return self.px_vol_ds

    @staticmethod
    def get_universe_px_vol(symbols, freq='1d'):
        """
        Returns full open, close, high, low, volume dataframe
        """
        super_list = []
        for n, t in tqdm(enumerate(symbols)):
            try:
                df = get_symbol_pricing(t, freq='1d', cols=None)
                df.drop_duplicates(inplace=True)
                df.index.name = 'storeDate'
                df['symbol'] = t
                df.set_index('symbol', append=True, inplace=True)
                super_list.append(df)
            except Exception as e:
                print(f'Exception get_mults_px_vol: {t}{e}')

        px_vol_df = pd.concat(super_list, axis=0)
        px_vol_df.drop_duplicates(inplace=True)
        index_cols = ['storeDate', 'symbol']
        px_vol_df = px_vol_df.reset_index().drop_duplicates(subset=index_cols).set_index(index_cols)
        return px_vol_df.unstack()

    @staticmethod
    def roll_vol(df, rw): return df.rolling(rw).std() * pow(252, 1/2)

    @staticmethod
    def get_df(ticker, desc, desc_df, period, tgt_df):
        if ticker in desc_df.index:
            return tgt_df[period][shorten_name(desc_df.loc[ticker, desc])]
        else: return np.nan

    @staticmethod
    def max_draw(xs):
        l_dd = np.argmax(np.maximum.accumulate(xs) - xs)
        h_dd = np.argmax(np.array(xs[:l_dd]))
        return xs[l_dd]/xs[h_dd]-1

    @staticmethod
    def max_pull(xs):
        h_p = np.argmax(xs - np.minimum.accumulate(xs))
        l_p = np.argmin(np.array(xs[:h_p]))
        return xs[h_p]/xs[l_p]-1

    @staticmethod
    def sign_compare(x, y):
        x_abs = np.abs(x)
        res = x_abs // y
        return (res * np.sign(x))

    @staticmethod
    def pct_of(df, count_df, name):
        df = count_df.T.count() / df.T.count()
        df.name = name
        return df

    @staticmethod
    def pct_above_series(df, key, tresh):
        count_df = df[df > tresh] if tresh >= 0 else df[df < tresh]
        return BaseDS.pct_of(df, count_df, key)

    @staticmethod
    def forward_returns(df, look_ahead, smooth=None):
        """ New forward returns, single period """
        if smooth is None:
            smooth = int(look_ahead/4)
        spct_chg = df.pct_change(look_ahead).rolling(smooth).mean()
        return spct_chg.shift(-int(smooth/2)).shift(-look_ahead)

    @staticmethod
    def discretize_returns(df, treshs, classes):
        """ discretize forward returns into classes """
        if isinstance(df, pd.Series):
            return pd.cut(df.dropna(), treshs, labels=classes)
        else:
            df.dropna(inplace=True)
            for c in df.columns:
                df[c] = pd.cut(df[c], treshs, labels=classes)
        return df

    @staticmethod
    def labelize_ycol(df, ycol_name, cut_range, labels):
        """ replaces numeric with labels for classification """
        df[ycol_name] = BaseDS.discretize_returns(
            df[ycol_name], cut_range, labels)
        df.dropna(subset=[ycol_name], inplace=True)
        df[ycol_name] = df[ycol_name].astype(str)
        print(pd.value_counts(df[ycol_name]) / pd.value_counts(df[ycol_name]).sum())
        new_order = excl(df.columns, [ycol_name]) + [ycol_name]
        df = df[new_order]