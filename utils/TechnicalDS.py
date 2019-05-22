
import os
import pandas as pd
import numpy as np
from utils.basic_utils import UNIVERSE
from utils.basic_utils import numeric_cols, excl
from utils.pricing import get_universe_px_vol
from utils.pricing import shorten_name
from utils.pricing import to_index_form, eq_wgt_indices

from utils.BaseDS import BaseDS


class TechnicalDS(BaseDS):

    def __init__(self,
        path='../tmp/',
        fname='universe-px-vol-ds.h5',
        load_ds=True,
        tickers=None, bench='^GSPC', look_ahead=120, look_back=252*7,
        invert_list=[], include_list=[],
        roll_vol_days=30,
        pct_chg_keys=[1, 20, 50, 200],
        quantile=0.75, max_draw_on=False):

        BaseDS.__init__(self, path, fname, load_ds,
            tickers, bench, look_ahead, look_back, quantile)

        self.invert_list = invert_list
        self.include_list = include_list
        self.pct_chg_keys = pct_chg_keys
        self.roll_vol_days = roll_vol_days
        self.max_draw_on = max_draw_on
        self.ycol_name = f'{self.y_col_name}{self.look_ahead}'

        self.sector_dict = self.dict_by_profile_column(
            self.tickers, 'sector', self.sectors)
        self.ind_dict = self.dict_by_profile_column(
            self.tickers, 'industry', self.industries)

        self.incl_feat_dict = None
        self.incl_group_feat_dict = None


    def load_px_vol_ds(self):
        """
        Refresh price and volume daily,
        used by most models with technical stats without refreshing
        """
        if os.path.isfile(self.path + self.fname) and self.load_ds:
            self.px_vol_ds = pd.read_hdf(self.path + self.fname, 'px_vol_df')
        else:
            # file does not exist, refreshes full dataset
            px_vol_ds = get_universe_px_vol(UNIVERSE)
            num_cols = numeric_cols(px_vol_ds)
            px_vol_ds.loc[:, num_cols] = px_vol_ds[num_cols].astype(np.float32)
            os.makedirs(self.path, exist_ok=True)
            px_vol_ds.to_hdf(self.path + self.fname, 'px_vol_df')
            # px_vol_ds.index = px_close.index.date
        return self.px_vol_ds

    def dict_by_profile_column(self, tickers, desc_col, subset):
        """ Maps companies to a descriptive column from profile """
        return {
            shorten_name(x):
            list(self.profile.loc[
                self.profile.index.isin(tickers) &
                self.profile[desc_col].isin([x]),:
            ].index) for x in subset
        }

    def create_base_frames(self):
        """
        Price and volume base dataframes, Adjusted for SP500 trading days
        """
        self.active_keys = self.pct_chg_keys[1:]
        self.incl_feat_dict = {}

        print('OCLHV dataframes')
        self.close_df = self.px_vol_df['close'].dropna(subset=[self.bench])
        self.open_df = self.px_vol_df['open'].dropna(subset=[self.bench])
        self.low_df = self.px_vol_df['low'].dropna(subset=[self.bench])
        self.high_df = self.px_vol_df['high'].dropna(subset=[self.bench])

        # inverted securities before transforms
        print('Inverting instruments')
        for df in (self.close_df, self.open_df, self.low_df, self.high_df):
            df[self.invert_list] = 1/df[self.invert_list]

        self.vol_df = self.px_vol_df['volume'].dropna(subset=[self.bench]) # relative to 10, 60 day average
        self.dollar_value_df = self.close_df * self.vol_df # relative to 10, 60 day average

        print('Change dataframes')
        self.close_1d_shift_df = self.close_df.shift(1) # closed shifted 1 day for calcs
        self.close_1d_chg_df = self.close_df - self.close_1d_shift_df # 1 day change
        self.pct_chg_df_dict = {x: self.close_df.pct_change(x) for x in self.pct_chg_keys}
        self.intra_day_chg_df = (self.close_df - self.open_df) / self.open_df # intra day range
        self.open_gap_df = (self.open_df - self.close_1d_shift_df) / self.close_1d_shift_df # open gap

        self.incl_feat_dict.update({'IntraDayChg': self.intra_day_chg_df})
        self.incl_feat_dict.update({'OpenGap': self.open_gap_df})

        print('Relative performance dataframes')
        # % of 50 day moving average
        self.pct_50d_ma_df = self.close_df / self.close_df.rolling(50).mean()
        # % of 200 day moving average
        self.pct_200d_ma_df = self.close_df / self.close_df.rolling(200).mean()
        # % of 52 week high
        self.pct_52wh_df = self.close_df / self.close_df.rolling(252).max()
        # % of 52 week low
        self.pct_52wl_df = self.close_df / self.close_df.rolling(252).min()

        self.incl_feat_dict.update({'Pct50MA': self.pct_50d_ma_df})
        self.incl_feat_dict.update({'Pct200MA': self.pct_200d_ma_df})
        self.incl_feat_dict.update({'Pct52WH': self.pct_52wh_df})
        self.incl_feat_dict.update({'Pct52WL': self.pct_52wl_df})

        print('Relative volume and dollar value dataframes')
        # vol as a pct of 10 day average
        self.pct_vol_10da_df = self.vol_df / self.vol_df.rolling(10).mean()
        # vol as a pct of 60 day average
        self.pct_vol_50da_df = self.vol_df / self.vol_df.rolling(50).mean()
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
            lambda x: TechnicalDS.roll_vol(x, self.roll_vol_days))
        self.incl_feat_dict.update({f'RollRealVol{self.roll_vol_days}': self.roll_realvol_df})

        print('Percentage change stds dataframes')
        self.pct_stds_df_dict = {
            x: self.pct_chg_df_dict[x].apply(lambda x: self.sign_compare(x, x.std()))
            for x in self.pct_chg_keys}

        print('Ranked returns dataframes')
        self.hist_perf_ranks = {
            k: self.close_df[self.companies]
            .apply(lambda x: (x.pct_change(k)+1))
            .apply(lambda x: x.rank(pct=True, ascending=True), axis=0)
            for k in self.active_keys}

        if self.max_draw_on:
            print(f'Max draw/pull dataframes')
            self.max_draw_df = self.close_df.rolling(self.look_ahead).apply(
                lambda x: self.max_draw(x), raw=True)
            self.max_pull_df = self.close_df.rolling(self.look_ahead).apply(
                lambda x: self.max_pull(x), raw=True)

            self.incl_feat_dict.update({f'MaxDraw{self.look_ahead}': self.max_draw_df})
            self.incl_feat_dict.update({f'MaxPull{self.look_ahead}': self.max_pull_df})

        print('Forward return dataframe')
        self.fwd_return_df = self.close_df.apply(lambda x:
            TechnicalDS.forward_returns(x, self.look_ahead))
        self.incl_feat_dict.update({self.ycol_name: self.fwd_return_df})

    def technical_transforms(self, symbol, incl_name=False, incl_close=False):
        """
        Create technical transformations for a single instrument
        Can be used for both micro and macro
        """
        if self.incl_feat_dict is None:
            self.create_base_frames()
        ndf = pd.DataFrame()
        pre = symbol if incl_name else ''
        if incl_close:
            ndf[f'{symbol}Close'] = self.close_df[symbol]
        for p in self.pct_chg_keys:
            ndf[f'{pre}PctChg{p}'] = self.pct_chg_df_dict[p][symbol]
        for p in self.pct_stds_df_dict.keys():
            ndf[f'{pre}PctChgStds{p}'] = self.pct_stds_df_dict[p][symbol]
        for d in self.incl_feat_dict.keys():
            ndf[f'{pre}{d}'] = self.incl_feat_dict[d][symbol]
        for p in self.active_keys:
            ndf[f'{pre}PerfRank{p}'] = self.hist_perf_ranks[p][symbol]

        return ndf

    def stitch_instruments(self, axis=0):
        """
        Stitch all companies vertically
        """
        if self.incl_feat_dict is None:
            self.create_base_frames()
        super_list = []
        for t in self.tickers:
            incl_close = True if t in self.invert_list else False
            df = self.technical_transforms(t, incl_name=False, incl_close=incl_close)
            df['symbol'] = t
            df.set_index('symbol', append=True, inplace=True)
            super_list.append(df)
        return pd.concat(super_list, axis=axis)

    def create_group_features(self):

        print('Group index')
        self.bench_index = to_index_form(self.close_df[self.bench], self.universe_key)
        self.sect_index = eq_wgt_indices(
            self.profile, self.close_df[self.companies],
            'sector', self.sectors, subset=self.companies)
        self.ind_index = eq_wgt_indices(
            self.profile, self.close_df[self.companies],
            'industry', self.industries, subset=self.companies)

        print('Group percentage changes')
        self.pct_chg_bench_dict = {x: self.bench_index.pct_change(x)
            for x in self.active_keys[1:]}
        self.pct_chg_sect_dict = {x: self.sect_index.pct_change(x)
            for x in self.active_keys[1:]}
        self.pct_chg_ind_dict = {x: self.ind_index.pct_change(x)
            for x in self.active_keys[1:]}

        print('Group pct stds')
        self.bench_pct_stds_df = {x: self.pct_chg_bench_dict[x].apply(
            lambda m: self.sign_compare(m, m.std()))
            for x in self.active_keys[1:]}
        self.sect_pct_stds_df = {x: self.pct_chg_sect_dict[x].apply(
            lambda m: self.sign_compare(m, m.std()))
            for x in self.active_keys[1:]}
        self.ind_pct_stds_df = {x: self.pct_chg_ind_dict[x].apply(
            lambda m: self.sign_compare(m, m.std()))
            for x in self.active_keys[1:]}

        print('Group performance deltas')
        self.bench_delta_dict = {k:
            self.pct_chg_df_dict[k][self.companies].subtract(
            self.pct_chg_bench_dict[k].values, axis=1)
                                 for k in self.active_keys[1:]}
        self.sect_delta_dict = {k:
            self.pct_chg_df_dict[k][self.companies].apply(
            lambda x: x - self.get_df(
                x.name, 'sector', self.profile,
                k, self.pct_chg_sect_dict)) for k in self.active_keys[1:]}
        self.ind_delta_dict = {k: self.pct_chg_df_dict[k][self.companies].apply(
            lambda x: x - self.get_df(
            x.name, 'industry', self.profile,
                k, self.pct_chg_ind_dict)) for k in self.active_keys[1:]}

        print('% above MA by group')
        self.pct_mt50ma_by_group_df = self.pct_above_tresh_by_group(
            self.pct_50d_ma_df, self.companies, 1)
        self.pct_mt200ma_by_group_df = self.pct_above_tresh_by_group(
            self.pct_200d_ma_df, self.companies, 1)

        print('% positive / negative chg stds by group')
        self.pct_pos_stds_by_group_df = self.pct_above_tresh_by_group(
            self.pct_stds_df_dict[50], self.companies, 0.99)
        self.pct_neg_stds_by_group_df = self.pct_above_tresh_by_group(
            self.pct_stds_df_dict[50], self.companies, -0.99)

        print('Group 50 day stds')
        self.group_50stds_df = pd.concat([
            self.bench_pct_stds_df[50],
            self.sect_pct_stds_df[50],
            self.ind_pct_stds_df[50]], axis=1)
        self.group_200stds_df = pd.concat([
            self.bench_pct_stds_df[200],
            self.sect_pct_stds_df[200],
            self.ind_pct_stds_df[200]], axis=1)

        print('Group 50 day deltas')
        self.group_50deltas_df = pd.concat([
            self.bench_delta_dict[50],
            self.sect_delta_dict[50],
            self.ind_delta_dict[50]], axis=1)
        self.group_200deltas_df = pd.concat([
            self.bench_delta_dict[200],
            self.sect_delta_dict[200],
            self.ind_delta_dict[200]], axis=1)

        print(f'Creating group dictionary')
        self.incl_group_feat_dict = {
            'pctGt50MA': self.pct_mt50ma_by_group_df,
            'pctGt200MA': self.pct_mt200ma_by_group_df,
            'pctPosStds': self.pct_pos_stds_by_group_df,
            'pctNegStds': self.pct_neg_stds_by_group_df,
            '50Stds': self.group_50stds_df,
            '200Stds': self.group_200stds_df,
        }

    def stitch_companies_groups(self):
        """ stitch all companies and group features """

        if self.incl_feat_dict is None:
            self.create_base_frames()
        if self.incl_group_feat_dict is None:
            self.create_group_features()

        super_list = []
        for t in self.tickers:
            # instrument features
            incl_close = True if t in self.invert_list else False
            inst_df = self.technical_transforms(
                t, incl_name=False, incl_close=incl_close)
            # group features
            symbol_list = []
            for key in self.incl_group_feat_dict.keys():
                if t in self.profile.index:
                    df = self.incl_group_feat_dict[key][self.get_group_keys(t)]
                    df.columns = [key + x for x in ['All', 'Sect', 'Ind']]
                    symbol_list.append(df)
            for x in {
                'delta50': self.group_50deltas_df,
                'detlta200': self.group_200deltas_df}.items():
                key, delta = x
                df = delta[t]
                df.columns = [key + x for x in ['All', 'Sect', 'Ind']]
                symbol_list.append(df)
            group_df = pd.concat(symbol_list, axis=1)
            symbol_df = pd.concat([inst_df, group_df], axis=1)
            symbol_df['symbol'] = t
            symbol_df.set_index('symbol', append=True, inplace=True)
            super_list.append(symbol_df)

        combined_df = pd.concat(super_list, axis=0, sort=False)
        print(f'Dataset columns: {combined_df.columns}')
        print(f'Dataset shape: {combined_df.shape}')

        return combined_df

    def get_group_keys(self, symbol):
        sect_ind = [shorten_name(x) for x in list(
            self.profile.loc[symbol, ['sector', 'industry']])]
        return [self.universe_key] + sect_ind

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
        x_abs = np.abs(x); res = x_abs // y
        return (res * np.sign(x))

    @staticmethod
    def pct_of(df, count_df, name):
        df = count_df.T.count() / df.T.count()
        df.name = name
        return df

    @staticmethod
    def pct_above_series(df, key, tresh):
        count_df = df[df > tresh] if tresh >= 0 else df[df < tresh]
        return TechnicalDS.pct_of(df, count_df, key)

    def pct_above_tresh_by_group(self, tgt_df, universe, tresh):
        """
        Percentage of companies above/below a treshold
        across universe, sectors, and industries
        """
        univ_df = self.pct_above_series(
            tgt_df[universe], self.universe_key, tresh)
        sect_df = pd.concat([
            self.pct_above_series(
                tgt_df[self.sector_dict[s]], s, tresh)
            for s in self.sector_dict.keys()], axis=1)
        ind_df = pd.concat([
            self.pct_above_series(
                tgt_df[self.ind_dict[s]], s, tresh)
            for s in self.ind_dict.keys()], axis=1)
        return pd.concat([univ_df, sect_df, ind_df], axis=1)

    def return_intervals(self, tresholds=[0.25, 0.75]):
        """ Used for discretizing historical returns into classes """
        px = self.fwd_return_df
        low_q = list(px.where(px < 0).mean().quantile(tresholds))
        high_q = list(px.where(px > 0).mean().quantile(tresholds))
        return (-np.inf, low_q[0], low_q[1], high_q[0], high_q[1], np.inf)

    @staticmethod
    def forward_returns(df, look_ahead, smooth=None):
        """ New forward returns, single period """
        if smooth is None: smooth = int(look_ahead/4)
        spct_chg = df.pct_change(look_ahead).rolling(smooth).mean()
        return spct_chg.shift(-int(smooth/2)).shift(-look_ahead)

    @staticmethod
    def discretize_returns(df, treshs, classes):
        """ discretize forward returns into classes """
        if isinstance(df, pd.Series): return pd.cut(df.dropna(), treshs, labels=classes)
        else:
            df.dropna(inplace=True)
            for c in df.columns: df[c] = pd.cut(df[c], treshs, labels=classes)
        return df

    @staticmethod
    def labelize_ycol(df, ycol_name, cut_range, labels):
        """ replaces numeric with labels for classification """
        df[ycol_name] = TechnicalDS.discretize_returns(
            df[ycol_name], cut_range, labels)
        df.dropna(subset=[ycol_name], inplace=True)
        df[ycol_name] = df[ycol_name].astype(str)
        new_order = excl(df.columns, ycol_name) + [ycol_name]
        df = df[new_order]
