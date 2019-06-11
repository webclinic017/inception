import logging, json, os
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, date
import pandas as pd
from utils.basic_utils import store_s3, load_s3, csv_store, csv_ext, numeric_cols, UNIVERSE
from io import BufferedReader, BufferedWriter, BytesIO
from utils.pricing import get_pricing, build_px_struct

tmp_folder = './tmp/'
ds_name = 'universe-px-vol-ds.h5'
s3_key = f'ds_yf/processed/{ds_name}'

def save_s3_local(s3_key, tgt_path):
    """ save an S3 file locally """
    file_content = load_s3(s3_key).read()
    with open(tgt_path, 'wb') as f:
        f.write(file_content)
    print(f'Stored: {tgt_path}')
save_s3_local(s3_key, ds_name)
px_vol_df = pd.read_hdf(ds_name, 'px_vol_df')
print(px_vol_df.info())
os.remove(ds_name)

def put_file_s3(src_path, s3_key):
    """ save a local file into S3 """
    file_content = None
    with open(src_path, 'r+b') as f: 
        file_content = f.read()
    store_s3(file_content, s3_key)
    print(f'Uploaded: {s3_key}')
# put_file_s3(f'{tmp_folder}{ds_name}', s3_key)

"""
# retrieve and create 
super_list = []
for t in UNIVERSE[:5]:
    try:
        print(f'Retrieving {t}')
        # store in S3
        data_dict = get_pricing(t, '1d', '15y')
        # process and append to list to speed up process
        df = build_px_struct(data_dict, '1d').dropna()
        df.drop_duplicates(inplace=True)
        df.index.name = 'storeDate'
        df['symbol'] = t
        df.set_index('symbol', append=True, inplace=True)
        super_list.append(df)
    except Exception as e:
        print(e)
# concatenating dataset
px_vol_df = pd.concat(super_list, axis=0)
px_vol_df.drop_duplicates(inplace=True)
index_cols = ['storeDate', 'symbol']
px_vol_df = px_vol_df.reset_index().drop_duplicates(subset=index_cols).set_index(index_cols)
px_vol_df = px_vol_df.unstack()
print(px_vol_df.info())
# convert to hd5
num_cols = numeric_cols(px_vol_df)
px_vol_df.loc[:, num_cols] = px_vol_df[num_cols].astype(np.float32)
# store locally
px_vol_df.to_hdf(ds_name, 'px_vol_df')
# upload to S3
put_file_s3(ds_name, s3_key)
# removing temp file
os.remove(ds_name)
"""

"""
# tmp_folder = './tmp/'
# file_content = None
# with open(f'{tmp_folder}AAPLReportsFinSummary.xml', 'r') as f:
#     file_content = f.read()
# s3_path = 'ib_ds/fundamental/reports/{}/'
# path = s3_path.format('ReportsFinSummary')
# fname = 'AAPL.xml'
# store_s3(file_content, path+fname)

# print(str(date.today()))]
"""