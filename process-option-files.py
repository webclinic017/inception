
# imports
from basic_utils import *

########################
# Testing / iteration
########################

dates = read_dates('option')
# predict_days = -1
# predict_dates, train_dates = dates[predict_days:], dates[:predict_days]
# print(predict_dates, train_dates)

# path = get_path('option', dates[0])
# result = json_load(path + 'AAPL.json')
# print(list_files('option', dates[0]))
#
# quotes = flatten_quotes([dates[0], '2018-09-18'])
# print(quotes.shape)
# tgt_date = train_dates[0]
# options = flatten_options([tgt_date])
# csv_store(options, get_path('option_consol'), csv_ext.format(tgt_date), True)

# consolidates all options in files by date
for d in dates:
    quotes = flatten_quotes([d])
    csv_store(quotes, get_path('quote_consol'), csv_ext.format(d), True)

# consolidates all options in files by date
for d in dates:
    options = flatten_options([d])
    csv_store(options, get_path('option_consol'), csv_ext.format(d), True)

# quote_frame = load_consol_quotes(dates[:1])
# print(quote_frame.storeDate.unique().shape, quote_frame.symbol.unique().shape)
#
# option_frame = load_consol_options(dates[:1])
# print(option_frame.shape, option_frame.lastTradeDate.unique().shape)
