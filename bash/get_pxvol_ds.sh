#!/bin/bash
inference_ip=52.32.87.161
tmp_path=../tmp/
px_vol_ds=universe-px-vol-ds.h5
scp -i ~/.ssh/qc_infra.pem ubuntu@$inference_ip:~/inception/tmp/$px_vol_ds $tmp_path$px_vol_ds
