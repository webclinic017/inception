
import pandas as pd
import numpy as np
from utils.basic_utils import config
from utils.pricing import shorten_name
from utils.pricing import to_index_form, eq_wgt_indices
from utils.fundamental import best_performers

from utils.BaseDS import BaseDS


class TechnicalDS(BaseDS):

    def __init__(
        self,
        path='../tmp/',
        fname='universe-px-vol-ds.h5',
        load_ds=True,
        tickers=None,
        bench='^GSPC',
        look_ahead=120, look_back=252*7,
        invert_list=[], include_list=[],
        roll_vol_days=30,
        pct_chg_keys=[1, 20, 50, 200],
        quantile=0.75, max_draw_on=False):

        BaseDS.__init__(self, path, fname, load_ds,
            bench, look_ahead, look_back, quantile)

        self.invert_list = invert_list
        self.include_list = include_list
        self.pct_chg_keys = pct_chg_keys
        self.roll_vol_days = roll_vol_days
        self.max_draw_on = max_draw_on
        self.active_keys = self.pct_chg_keys[1:]
        self.ycol_name = f'{self.y_col_name}{self.look_ahead}'
        self.companies = config['companies']

        if tickers is None:
            self.tickers = list(best_performers(
                self.clean_px, self.companies,
                self.look_back, self.quantile).index)
        elif tickers == 'All':
            self.tickers = self.companies
            print(f'{len(self.companies)} companies')
        else:
            self.tickers = tickers

        self.sectors = self.profile.loc[
            self.profile.symbol.isin(self.companies)].sector.unique()
        self.industries = self.profile.loc[
            self.profile.symbol.isin(self.companies)].industry.unique()
        print(f'Universe sectors: {self.sectors.shape[0]}, industries: {self.industries.shape[0]}')

        self.sector_dict = self.dict_by_profile_column(
            self.tickers, 'sector', self.sectors)
        self.ind_dict = self.dict_by_profile_column(
            self.tickers, 'industry', self.industries)

        self.incl_feat_dict = None
        self.incl_group_feat_dict = None

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
        self.vol_df = self.px_vol_df['volume'].dropna(subset=[self.bench])
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

        print('Relative performance dataframes')
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
            for x in self.pct_chg_keys
            }

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
            lambda x: BaseDS.forward_returns(x, self.look_ahead)
            )
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
        for d in self.incl_feat_dict.keys():
            df = self.incl_feat_dict[d]
            # print(d, symbol, fwd_symbol, d == self.ycol_name 
            # and symbol is not fwd_symbol)
            if symbol in df.columns:
                ndf[f'{pre}{d}'] = df[symbol]

        return ndf

    def stitch_instruments(self, symbols=None, name=False, axis=0):
        """
        Stitch all companies vertically
        """
        if self.incl_feat_dict is None:
            self.create_base_frames()
        if symbols is None:
            symbols = self.tickers
        super_list = []
        for t in symbols:
            incl_close = True if t in self.include_list else False
            fwd_ret_symbol = t if axis == 0 else self.bench
            df = self.technical_transforms(t, incl_name=name, incl_close=incl_close)
            if axis == 0:
                df['symbol'] = t
                df.set_index('symbol', append=True, inplace=True)
            super_list.append(df)
        return pd.concat(super_list, axis=axis)

    def create_company_features(self):

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
            lambda m: BaseDS.sign_compare(m, m.std()))
            for x in self.active_keys[1:]}
        self.sect_pct_stds_df = {x: self.pct_chg_sect_dict[x].apply(
            lambda m: BaseDS.sign_compare(m, m.std()))
            for x in self.active_keys[1:]}
        self.ind_pct_stds_df = {x: self.pct_chg_ind_dict[x].apply(
            lambda m: BaseDS.sign_compare(m, m.std()))
            for x in self.active_keys[1:]}

        self.incl_group_feat_dict = {}

        for k in self.ind_pct_stds_df.keys():
            self.incl_group_feat_dict.update(
            {f'{k}Stds':pd.concat([
                self.bench_pct_stds_df[k],
                self.sect_pct_stds_df[k],
                self.ind_pct_stds_df[k]], axis=1)}
            )

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

        self.incl_group_feat_dict.update({'pctGt50MA': self.pct_mt50ma_by_group_df})
        self.incl_group_feat_dict.update({'pctGt200MA': self.pct_mt200ma_by_group_df})

        print('% positive / negative chg stds by group')
        self.pct_pos_stds_by_group_df = self.pct_above_tresh_by_group(
            self.pct_stds_df_dict[50], self.companies, 0.99)
        self.pct_neg_stds_by_group_df = self.pct_above_tresh_by_group(
            self.pct_stds_df_dict[50], self.companies, -0.99)

        self.incl_group_feat_dict.update({
            'pctPosStds': self.pct_pos_stds_by_group_df
            })
        self.incl_group_feat_dict.update({
            'pctNegStds': self.pct_neg_stds_by_group_df
            })

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

        self.group_idx = pd.concat([
            self.bench_index,
            self.sect_index,
            self.ind_index], axis=1)

        print('Ranked returns dataframes')
        self.hist_perf_ranks = {
            k: self.group_idx
            .apply(lambda x: (x.pct_change(k)+1))
            .apply(lambda x: x.rank(pct=True, ascending=True), axis=0)
            for k in self.active_keys}

        for p in self.hist_perf_ranks.keys():
            self.incl_group_feat_dict.update({
                f'PerfRank{p}': self.hist_perf_ranks[p]
                })

        # self.incl_group_feat_dict = {
        #     'pctGt50MA': self.pct_mt50ma_by_group_df,
        #     'pctGt200MA': self.pct_mt200ma_by_group_df,
        #     '50Stds': self.group_50stds_df,
        #     '200Stds': self.group_200stds_df,
        # }

    def stitch_companies_groups(self):
        """ stitch all companies and group features """

        if self.incl_feat_dict is None:
            self.create_base_frames()
        if self.incl_group_feat_dict is None:
            self.create_company_features()

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

    def return_intervals(self, tresholds=[0.5, 0.75]):
        """ Used for discretizing historical returns into classes """
        px = self.fwd_return_df
        npa = px.values.reshape(-1,)
        npa = npa[~np.isnan(npa)]
        high_q = np.quantile(npa[np.where(npa > 0)], tresholds)
        low_q = np.quantile(
            npa[np.where(npa < 0)], list(1 - np.array(tresholds[::-1])))
        cuts = (-np.inf, low_q[0], low_q[1], 0, high_q[0], high_q[1], np.inf)
        print(f'Treshold distributions: {np.round(cuts, 2)}')
        return cuts

    def get_group_keys(self, symbol):
        sect_ind = [shorten_name(x) for x in list(
            self.profile.loc[symbol, ['sector', 'industry']])]
        return [self.universe_key] + sect_ind