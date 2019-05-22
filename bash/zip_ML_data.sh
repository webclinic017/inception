#!/bin/bash
p=/home/$(whoami)/inception
inference_IP=52.13.205.29
models_fname=all_models.tar.gz
tmp_fname=all_tmp.tar.gz

echo '>> Zipping models'
cd $p/ML
rm -f $models_fname
tar -cvzf $models_fname .

echo '>> Zipping temp files'
cd $p/tmp
rm -f $tmp_fname
tar -cvzf $tmp_fname .

echo '>> Moving files to inference instance'
scp -i ~ubuntu/.ssh/qc_infra.pem $p/ML/$models_fname ubuntu@$inference_IP:$p/ML/$models_fname
scp -i ~ubuntu/.ssh/qc_infra.pem $p/tmp/$tmp_fname ubuntu@$inference_IP:$p/tmp/$tmp_fname

cd $p
