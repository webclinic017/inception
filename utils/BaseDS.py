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
        bench=None,
        look_ahead=120,
        fwd_smooth=None,
        look_back=252*7,
        invert_list=[], 
        include_list=[],        
        quantile=0.75,
        max_draw_on=False
    ):

        self.path = path
        self.fname = fname
        self.load_ds = load_ds
        self.bench =  bench if len(bench) > 1 else [bench]
        self.look_ahead = look_ahead
        self.fwd_smooth = fwd_smooth
        self.look_back = look_back
        self.invert_list = invert_list
        self.include_list = include_list        
        self.max_draw_on = max_draw_on
        self.ycol_name = f'{self.y_col_name}{self.look_ahead}'
        self.quantile = quantile
        self.universe_dict = {k: config[k] for k in config['universe_list']}

        self.px_vol_df = self.load_px_vol_ds()
        self.px_vol_df = self.px_vol_df.tail(self.look_back)
        self.clean_px = self.px_vol_df['close'].dropna(subset=self.bench)

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

    def create_base_frames(self):
        """
        Price and volume base dataframes, Adjusted for SP500 trading days
        """
        self.incl_feat_dict = {}

        print('OCLHV dataframes')
        self.close_df = self.px_vol_df['close'].dropna(subset=self.bench)
        self.open_df = self.px_vol_df['open'].dropna(subset=self.bench)
        self.low_df = self.px_vol_df['low'].dropna(subset=self.bench)
        self.high_df = self.px_vol_df['high'].dropna(subset=self.bench)

        # inverted securities before transforms
        print('Inverted instruments')
        for df in (self.close_df, self.open_df, self.low_df, self.high_df):
            df[self.invert_list] = 1/df[self.invert_list]
        self.vol_df = self.px_vol_df['volume'].dropna(subset=self.bench)
        self.dollar_value_df = self.close_df * self.vol_df

        print('Change dataframes')
        self.close_1d_shift_df = self.close_df.shift(1)
        self.close_1d_chg_df = self.close_df - self.close_1d_shift_df
        self.pct_chg_df_dict = {x: self.close_df.pct_change(x) for x in self.pct_chg_keys}
        self.intra_day_chg_df = (self.close_df - self.open_df) / self.open_df
        self.open_gap_df = (self.open_df - self.close_1d_shift_df) / self.close_1d_shift_df

        for p in self.pct_chg_df_dict.keys():
            self.incl_feat_dict.update({f'PctChg{p}': self.pct_chg_df_dict[p]})

        self.incl_feat_dict.update({'IntraDayChg': self.intra_day_chg_df})
        self.incl_feat_dict.update({'OpenGap': self.open_gap_df})

        print('Moving averages/52Wk percentage dataframes')
        # % of 50 day moving average
        self.pct_50d_ma_df = self.close_df / self.close_df.fillna(method='ffill').rolling(50).mean()
        # % of 200 day moving average
        self.pct_200d_ma_df = self.close_df / self.close_df.fillna(method='ffill').rolling(200).mean()
        # % of 52 week high
        self.pct_52wh_df = self.close_df / self.close_df.fillna(method='ffill').rolling(252).max()
        # % of 52 week low
        self.pct_52wl_df = self.close_df / self.close_df.fillna(method='ffill').rolling(252).min()

        self.incl_feat_dict.update({'Pct50MA': self.pct_50d_ma_df})
        self.incl_feat_dict.update({'Pct200MA': self.pct_200d_ma_df})
        self.incl_feat_dict.update({'Pct52WH': self.pct_52wh_df})
        self.incl_feat_dict.update({'Pct52WL': self.pct_52wl_df})

        print('Relative volume and dollar value dataframes')
        # vol as a pct of 10 day average
        self.pct_vol_10da_df = self.vol_df / self.vol_df.fillna(method='ffill').rolling(10).mean()
        # vol as a pct of 60 day average
        self.pct_vol_50da_df = self.vol_df / self.vol_df.fillna(method='ffill').rolling(50).mean()
        # dollar values % of 10 day ma
        self.pct_dv_10da_df = self.dollar_value_df / self.dollar_value_df.rolling(10).mean()
        # dollar values % of 50 day ma
        self.pct_dv_50da_df = self.dollar_value_df / self.dollar_value_df.rolling(50).mean()

        self.incl_feat_dict.update({'PctVol10DA': self.pct_vol_10da_df})
        self.incl_feat_dict.update({'PctVol50DA': self.pct_vol_50da_df})
        self.incl_feat_dict.update({'PctDV10DA': self.pct_dv_10da_df})
        self.incl_feat_dict.update({'PctDV50DA': self.pct_dv_50da_df})

        print('Realized volatility dataframe')
        # 30 day rolling daily realized return volatility
        self.roll_realvol_df = self.pct_chg_df_dict[1].apply(
            lambda x: BaseDS.roll_vol(x, self.roll_vol_days))
        self.incl_feat_dict.update({f'RollRealVol{self.roll_vol_days}': self.roll_realvol_df})

        print('Percentage change stds dataframes')
        self.pct_stds_df_dict = {
            x: self.pct_chg_df_dict[x].apply(lambda x: BaseDS.sign_compare(x, x.std()))
            for x in self.pct_chg_keys}

        for p in self.pct_stds_df_dict.keys():
            self.incl_feat_dict.update({f'PctChgStds{p}': self.pct_stds_df_dict[p]})

        if self.max_draw_on:
            print(f'Max draw/pull dataframes')
            self.max_draw_df = self.close_df.rolling(self.look_ahead).apply(
                lambda x: BaseDS.max_draw(x), raw=True)
            self.max_pull_df = self.close_df.rolling(self.look_ahead).apply(
                lambda x: BaseDS.max_pull(x), raw=True)

            self.incl_feat_dict.update({f'MaxDraw{self.look_ahead}': self.max_draw_df})
            self.incl_feat_dict.update({f'MaxPull{self.look_ahead}': self.max_pull_df})

        print('Ranked returns dataframes')
        self.hist_perf_ranks = {
            k: self.close_df[self.tickers]
            .apply(lambda x: (x.pct_change(k)+1))
            .apply(lambda x: x.rank(pct=True, ascending=True), axis=0)
            for k in self.active_keys}

        for p in self.hist_perf_ranks.keys():
            self.incl_feat_dict.update({
                f'PerfRank{p}': self.hist_perf_ranks[p]
                })

        print('Forward return dataframe')
        self.fwd_return_df = self.close_df.apply(
            lambda x: BaseDS.forward_returns(x, self.look_ahead, self.fwd_smooth))
        self.incl_feat_dict.update({self.ycol_name: self.fwd_return_df})

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