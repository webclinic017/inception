
from utils.basic_utils import config
from utils.basic_utils import read_dates, load_csvs, numeric_cols, excl
from utils.fundamental import best_performers

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

        self.px_vol_df = self.load_px_vol_ds()
        self.clean_px = self.px_vol_df['close'].dropna(subset=[self.bench])
        self.companies = config['companies']

        if tickers is None:
            self.tickers = list(best_performers(
                self.clean_px, self.companies,
                self.look_back, self.quantile).index)
        else:
            self.tickers = tickers

        # Quotes, profile, and industries
        self.dates = read_dates('quote')
        # last date saved in S3
        self.tgt_date = self.dates[-1]
        print(f'Target date: {self.tgt_date}')

        quotes = load_csvs('quote_consol', [self.tgt_date])
        quotes = quotes.loc[quotes.symbol.isin(self.companies)]
        self.quotes = quotes.set_index('symbol', drop=False)
        profile = load_csvs('summary_detail', ['assetProfile'])
        profile = profile.loc[profile.symbol.isin(self.companies)]
        self.profile = profile.set_index('symbol', drop=False)

        self.sectors = profile.loc[
            profile.symbol.isin(self.companies)].sector.unique()
        self.industries = profile.loc[
            profile.symbol.isin(self.companies)].industry.unique()
        print(f'Sectors: {self.sectors.shape[0]}, Industries: {self.industries.shape[0]}')
