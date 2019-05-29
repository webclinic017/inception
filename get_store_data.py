import sys
from utils.basic_utils import *
from utils.unpack_summary import *
from utils.pricing import load_px_close, get_pricing
from utils.BaseDS import BaseDS

if __name__ == '__main__':
    hook = sys.argv[1]
    today_date = str(date.today())

    if hook == 'quotes':
        get_quotes(UNIVERSE)
        quotes = flatten_quotes([str(today_date)])
        csv_store(quotes, get_path('quote_consol'), csv_ext.format(today_date), True)
    elif hook == 'summary':
        sample_set = config['companies'] + config['sectors'] +\
            config['bonds'] + config['risk']
        for t in sample_set:
            try:
                print('Retrieving {0} for {1}'.format(hook, t))
                get_grouped_ds(t, 'summary')
            except Exception as e:
                print(e)
        unpack_summaries([today_date])
    elif hook == 'pricing':
        for t in UNIVERSE:
            try:
                print('Retrieving {0} for {1}'.format(hook, t))
                # get_pricing(t, '1m', '5d')
                # get_pricing(t, '1d', '15y')
            except Exception as e:
                print(e)
        # persist closing price for current day
        temp_path, px_close_fname, px_vol_fname = 'tmp/', 'universe-px-ds', 'universe-px-vol-ds.h5'
        # print('Persisting pricing to {}'.format(temp_path + px_close_fname))
        # load_px_close(temp_path, px_close_fname, False)
        print(f'Persisting universe price and volume to {temp_path + px_vol_fname}')
        # baseDs = BaseDS(path=temp_path, fname=px_vol_fname, load_ds=False, )
        baseDs = BaseDS(path=temp_path, fname=px_vol_fname, load_ds=True, )
        px_close = baseDs.px_vol_df['close']
        num_cols = numeric_cols(px_close)
        px_close.loc[:, num_cols] = px_close[num_cols].astype(np.float32)
        os.makedirs(temp_path, exist_ok=True)
        px_close.to_parquet(path + fname)

    elif hook == 'options':
        sample_set = sys.argv[2]
        for t in config[sample_set]:
            try:
                print('Retrieving {0} for {1}'.format(hook, t))
                get_options(t)
            except Exception as e:
                print(e)
        options = flatten_options([today_date])
        csv_store(options, get_path('option_consol'), csv_ext.format(today_date), True)
    else:
        print('Please enter a valid option')
        print('Valid options are: quotes, summary, pricing, and options')
