#!/bin/bash
p=/home/$(whoami)/inception
models_fname=all_models.tar.gz
tmp_fname=all_tmp.tar.gz

echo '>> Unzip models'
cd $p/ML
tar -xzf $model_fname

echo '>> Unzip temp files'
cd $p/tmp
tar -xzf $tmp_fname

cd $p

