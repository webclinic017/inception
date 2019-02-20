import sys
from basic_utils import *
from unpack_summary import *

if __name__ == '__main__':
    hook = sys.argv[1]
    today_date = str(date.today())

    if hook == 'quotes':
        get_quotes(UNIVERSE)
        quotes = flatten_quotes([str(today_date)])
        csv_store(quotes, get_path('quote_consol'), csv_ext.format(today_date), True)
    elif hook == 'summary':
        for t in UNIVERSE:
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
                get_pricing(t, '1m', '5d')
                get_pricing(t, '1d', '10y')
            except Exception as e:
                print(e)
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
