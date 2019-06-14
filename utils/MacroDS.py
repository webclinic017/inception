
from utils.basic_utils import config

from utils.TechnicalDS import TechnicalDS
from utils.BaseDS import BaseDS


class MacroDS(TechnicalDS):

    def __init__(
        self,
        path='../tmp/',
        fname='universe-px-vol-ds.h5',
        load_ds=True,
        tickers=None, 
        bench='^GSPC', 
        look_ahead=60, 
        fwd_smooth=None,
        look_back=252,
        invert_list=[], 
        include_list=[],
        roll_vol_days=30,
        pct_chg_keys=[1, 20, 50, 200],
        quantile=0.75,
        max_draw_on=False
    ):

        BaseDS.__init__(
            self, path, fname, load_ds, bench, look_ahead, fwd_smooth, look_back, 
            invert_list, include_list, quantile, max_draw_on)

        instruments = list(config['universe_list'])
        instruments.remove('companies')
        self.universe_dict = {k: config[k] for k in instruments}

        if tickers is None:
            self.tickers = []
            [self.tickers.extend(self.universe_dict[k])
                for k in self.universe_dict.keys()]
        else:
            self.tickers = tickers

        self.invert_list = invert_list
        self.include_list = include_list
        self.pct_chg_keys = pct_chg_keys
        self.roll_vol_days = roll_vol_days
        self.max_draw_on = max_draw_on
        self.active_keys = self.pct_chg_keys[1:]
        self.ycol_name = f'{self.y_col_name}{self.look_ahead}'

        self.incl_feat_dict = None
        TechnicalDS.create_base_frames(self)
        self.fwd_return_df = self.fwd_return_df[[self.bench]]
        self.incl_feat_dict.update(
            {self.ycol_name: self.fwd_return_df[[self.bench]]})
        self.incl_feat_dict.update(
            {self.ycol_name: self.fwd_return_df[[self.bench]]})
