#$/bin/bash
python train.py --lr 0.075 --momentum 0.075 --num_hidden 3 --sizes 100,100,100 --activation sigmoid --loss ce --opt gd --batch_size 20 --anneal true --save_dir pa1/ --expt_dir pa1/exp1/ --train train.csv --test test.csv --val val.csv --pretrain false
