#!/bin/bash
p=/home/$(whoami)/inception
inference_IP=34.221.106.35
models_fname=all_models.tar.gz
tmp_fname=all_tmp.tar.gz

echo '>> Tar models'
cd $p/ML
rm -f $model_fname
tar -cvzf $model_fname *NN*

echo '>> Tar temp files'
cd $p/tmp
rm -f $tmp_fname
tar -cvzf $tmp_fname .

echo '>> Moving files to inference instance'
scp -i ~ubuntu/.ssh/qc_infra.pem $p/ML/$models_fname ubuntu@$inference_IP:$p/ML/$models_fname
scp -i ~ubuntu/.ssh/qc_infra.pem $p/tmp/$tmp_fname ubuntu@$inference_IP:$p/tmp/$tmp_fname

cd $p
