import sys
from basic_utils import *
from option_recommendations import *

if __name__ == '__main__':
    hook = sys.argv[1]
    today_date = str(date.today())
    if hook == 'quotes':
        get_quotes(UNIVERSE)
        quotes = flatten_quotes([str(today_date)])
        csv_store(quotes, get_path('quote_consol'), csv_ext.format(today_date), True)
    elif hook == 'summary':
        for t in UNIVERSE: get_grouped_ds(t, 'summary')
    elif hook == 'options':
        for t in UNIVERSE: get_options(t)
        options = flatten_options([today_date])
        csv_store(options, get_path('option_consol'), csv_ext.format(today_date), True)
        run_options_recommendation()
    else:
        print('Please enter a valid option')
        print('Valid options are: quotes, summary, and options')
