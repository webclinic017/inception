#!/bin/bash
p=/home/$(whoami)/inception
models_fname=all_models.tar.gz
tmp_fname=all_tmp.tar.gz

echo '>> Unzipping models'
cd $p/ML
tar xvzf $models_fname

echo '>> Unzipping temp files'
cd $p/tmp
tar xvzf $tmp_fname

cd $p
