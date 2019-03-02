import json
import random
import time
from io import StringIO
from datetime import datetime
from datetime import date
import urllib
from urllib.request import urlopen

import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

import boto3

# local save and load
def save_config(config, fname):
    with open(fname, 'w') as file:
        data = json.dumps(config, indent=1)
        file.write(data)
        file.close()
        print('Saving', fname)


def load_config(fname):
    with open(fname, 'r') as file:
        data = file.read()
        file.close()
        print('Loading', fname)
        return json.loads(data)

# url requests timers to prevent abuse
def run_sleeper(min_s, max_s):
    sleep_time = random.randint(min_s, max_s)
    time.sleep(sleep_time)

# clean up function
def clean_up_fmt(df):
    fmt_cols = [x for x in df.columns.tolist() if str.lower(x).find('fmt') > 0]
    raw_cols = [x for x in df.columns.tolist() if str.lower(x).find('raw') > 0]
    rndm_map = {x: x[:x.index('.')] for x in raw_cols}
    df.drop(fmt_cols, axis=1, inplace=True)
    df.rename(rndm_map, axis=1, inplace=True)
    return df

# S3 load and store
load_s3 = lambda f: bucket.Object(key=f).get()["Body"]
store_s3 = lambda obj, f: bucket.put_object(Body=obj, Key=f)
json_load = lambda f: json.load(load_s3(f))
csv_load = lambda f: StringIO(load_s3(csv_ext.format(f)).read().decode('utf-8'))

def csv_store(df, path, fname, headings=True):
    f = path + fname
    buffer = StringIO()
    df.to_csv(buffer, index=False, encoding='utf-8', header=headings)
    store_s3(buffer.getvalue(), f)
    print('Saved', f)

def read_dates(dataset):
    ds_list = [x.key for x in bucket.objects.filter(Prefix=dataset)]
    ds_list = list(
        set([x.split('/')[1][:10] for x in ds_list if x.find('.json') > 0])
    )
    return sorted(ds_list)

def get_path(dataset, d=None): return config[dataset + '_path'].format(d)
def list_files(dataset, date=None):
    return [x.key for x in bucket.objects.filter(Prefix=get_path(dataset, date))]

# data processing functions
def flatten_quotes(dates):
    quote_frame = pd.DataFrame()
    for d in dates:
        storeDate = datetime.strptime(d, '%Y-%m-%d').timestamp()
        path = get_path('quote')
        result = json_load(path + json_ext.format(d))
        for q in result:
            q_clean = clean_up_fmt(json_normalize(q))
            q_clean['storeDate'] = storeDate
            quote_frame = quote_frame.append(q_clean, sort=False)
    return quote_frame

def flatten_options(dates):
    calls_df = pd.DataFrame()
    puts_frame = pd.DataFrame()
    for d in dates:
        print('Working on {}'.format(d))
        storeDate = datetime.strptime(d, '%Y-%m-%d').timestamp()
        files = list_files('option', d)
        for f in files:
            print('Flattening options for {}'.format(f))
            option_expirations = json_load(f)
            updt_root_flag = True
            for expiration in option_expirations:
                if updt_root_flag:
                    underlyingSymbol = expiration['underlyingSymbol']
                    updt_root_flag = False
                options = expiration['options'][0]
                call_df = clean_up_fmt(json_normalize(options['calls']))
                put_df = clean_up_fmt(json_normalize(options['puts']))
                call_df['underlyingSymbol'] = underlyingSymbol
                call_df['storeDate'] = storeDate
                put_df['underlyingSymbol'] = underlyingSymbol
                put_df['storeDate'] = storeDate
                calls_df = calls_df.append(call_df, sort=False)
                puts_frame = puts_frame.append(put_df, sort=False)
    calls_df['type'] = 'call'
    puts_frame['type'] = 'put'
    full_set = calls_df.append(puts_frame)
    return full_set


def load_csvs(path_key, dates):
    df = pd.DataFrame()
    for d in dates:
        path = get_path(path_key)
        print('Loading file', path + d)
        result = csv_load(path + d)
        d_df = pd.read_csv(result)
        df = df.append(d_df, sort=False)
    return df


def load_consol_quotes(dates):
    quote_frame = pd.DataFrame()
    for d in dates:
        print('Loading quotes for', d)
        path = get_path('quote_consol')
        result = csv_load(path + d)
        quotes = pd.read_csv(result)
        quote_frame = quote_frame.append(quotes, sort=False)
    return quote_frame


def load_consol_options(dates):
    option_frame = pd.DataFrame()
    for d in dates:
        print('Loading options for', d)
        path = get_path('option_consol')
        result = csv_load(path + d)
        options = pd.read_csv(result)
        option_frame = option_frame.append(options, sort=False)
    return option_frame


# basic data gathering functions
def comma_join(x, y): return x + ',' + y
def get_children_list(ds, parent): return ds[parent]['result']
def url_open(url):
    run_sleeper(MIN_MAX_SLEEP[0], MIN_MAX_SLEEP[1])
    usock = urlopen(url)
    data = usock.read()
    usock.close()
    return data


def get_data(symbol, dataset, encoded_params):
    url = QUERY_DICT[dataset][url_key]
    data = url_open(url.format(symbol, encoded_params))
    return data

def get_data_params(dataset, param_dict):
    url = QUERY_DICT[dataset][url_key]
    data = url_open(url.format(**param_dict))
    return data

def get_grouped_ds(symbol, dataset):
    # bulks download of all description modules
    # streamline to avoid request information that does not change often
    key, value = QUERY_DICT[dataset][enc_key], QUERY_DICT[dataset][enc_val]
    encoded_kv = urllib.parse.urlencode({key: value})
    data = get_data(symbol, dataset, encoded_kv)
    full_data = get_children_list(json.loads(data), 'quoteSummary')
    today_date = str(date.today())
    data = json.dumps(full_data)
    path = get_path(dataset, today_date)
    store_s3(data, path + json_ext.format(symbol))


def get_cs_tickers(ticker_list):
    cs_tickers = ticker_list[0]
    if len(ticker_list) > 1:
        for t in ticker_list[1:]:
            cs_tickers = comma_join(cs_tickers, t)
    return cs_tickers


def get_quotes(symbol_list):
    dataset = 'quote'
    full_data = []
    index, max_elems = 0, MAX_SYMBOLS
    print('Getting quotes for', symbol_list)
    for q in range(int(len(symbol_list) / max_elems) + 1):
        subset = symbol_list[index:index + max_elems]
        index += max_elems
        symbols = get_cs_tickers(subset)
        encoded_kv = urllib.parse.urlencode(
            {QUERY_DICT[dataset][enc_key]: symbols}
        )
        try:
            data = get_data(encoded_kv, dataset, '')
            full_data.extend(get_children_list(json.loads(data), 'quoteResponse'))
        except Exception as e:
            print(e)
    data = json.dumps(full_data)
    path = get_path(dataset)
    store_s3(data, path + json_ext.format(str(today_date)))


def get_options(symbol):
    # save all options expirations dates to files for a given company
    dataset = 'option'
    print('Getting options expirations for', symbol)
    key = QUERY_DICT[dataset][enc_key]
    encoded_kv = urllib.parse.urlencode({key: 0})
    # first expiration no date
    data = get_data(symbol, dataset, encoded_kv)
    json_dict = json.loads(data)
    option_chain = json_dict['optionChain']['result'][0]
    exp_dates = option_chain['expirationDates']
    full_data = []
    for ed in exp_dates:
        encoded_kv = urllib.parse.urlencode({QUERY_DICT[dataset][enc_key]: ed})
        updateFmt = 'Downloading options the {0} expiration'
        print(updateFmt.format(str(date.fromtimestamp(ed))))
        data = get_data(symbol, dataset, encoded_kv)
        full_data.extend(get_children_list(json.loads(data), 'optionChain'))
    data = json.dumps(full_data)
    path = get_path(dataset, str(today_date))
    store_s3(data, path + json_ext.format(symbol))

def get_pricing(symbol, interval='1d', prange='5y'):
    # save pricing for a given interval and range
    dataset = 'pricing'
    point = {'s':symbol,'i':interval, 'r': prange}
    print('Getting pricing interval of {s} interval: {i}, range: {r}'.format(**point))
    # first expiration no date
    data = get_data_params('pricing', point)
    json_dict = json.loads(data)
    pricing_data = json_dict['chart']['result'][0]
    data = json.dumps(pricing_data)
    path = get_path(dataset, interval)
    store_s3(data, path + json_ext.format(symbol))
    return pricing_data

###### environment variables ######

config = load_config('utils/config.json')
QUERY_DICT = config["query_dict"]
COMPANIES = config['companies']
UNIVERSE = []
[UNIVERSE.extend(config[y]) for y in [z for z in config['universe_list']]]
MIN_MAX_SLEEP = config["min_max_sleep"]
MAX_SYMBOLS = config["max_symbols_request"]
S3_STORE = config['s3_store']
fname = config['filename_fmt']

today_date = date.today()

s3 = boto3.resource('s3', 'us-west-2')
BUCKET_NAME = config['bucket_name']
bucket = s3.Bucket(BUCKET_NAME)

url_key = 'url'
enc_key = 'enc_key'
enc_val = 'enc_val'
csv_ext = '{}.csv'
json_ext = '{}.json'
