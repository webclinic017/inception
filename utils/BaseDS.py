import os
from tqdm import *
from utils.basic_utils import load_csvs, config, UNIVERSE
from utils.basic_utils import read_dates, numeric_cols
from utils.pricing import get_symbol_pricing
import pandas as pd
import numpy as np

class BaseDS(object):

    universe_key = '^ALL'
    y_col_name = 'fwdRet'
    forward_return_labels = ["bear", "short", "neutral", "long", "bull"]

    def __init__(self,
        path='../tmp/',
        fname='universe-px-vol-ds.h5',
        load_ds=True, tickers=None,
        bench='^GSPC',
        look_ahead=120, look_back=252*7,
        quantile=0.75):

        self.path = path
        self.fname = fname
        self.load_ds = load_ds
        self.tickers = tickers
        self.bench = bench
        self.look_ahead = look_ahead
        self.look_back = look_back
        self.quantile = quantile
        self.universe_dict = {k: config[k] for k in config['universe_list']}

        self.px_vol_df = self.load_px_vol_ds()
        self.clean_px = self.px_vol_df['close'].dropna(subset=[self.bench])

        # Quotes, profile, and industries
        self.dates = read_dates('quote')
        # last date saved in S3
        self.tgt_date = self.dates[-1]
        print(f'Target date: {self.tgt_date}')

        quotes = load_csvs('quote_consol', [self.tgt_date])
        # quotes = quotes.loc[quotes.symbol.isin(self.companies)]
        self.quotes = quotes.set_index('symbol', drop=False)
        profile = load_csvs('summary_detail', ['assetProfile'])
        # profile = profile.loc[profile.symbol.isin(self.companies)]
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
            num_cols = numeric_cols(px_vol_ds)
            self.px_vol_ds.loc[:, num_cols] = px_vol_ds[num_cols].astype(np.float32)
            os.makedirs(self.path, exist_ok=True)
            self.px_vol_ds.to_hdf(self.path + self.fname, 'px_vol_df')
            # px_vol_ds.index = px_close.index.date
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
        return px_vol_df.unstack()
