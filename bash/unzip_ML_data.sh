#!/bin/bash
p=/home/$(whoami)/inception
models_fname=all_models.tar.gz
tmp_fname=all_tmp.tar.gz

echo '>> Unzip models'
cd $p/ML
tar xvzf $model_fname

echo '>> Unzip temp files'
cd $p/tmp
tar xvzf $tmp_fname

cd $p
