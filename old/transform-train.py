from basic_utils import *

dates = read_dates('option')
predict_days = -1
predict_dates, train_dates = dates[predict_days:], dates[:predict_days]

train_quotes = load_consol_quotes(train_dates)
train_options = load_consol_options(train_dates)
