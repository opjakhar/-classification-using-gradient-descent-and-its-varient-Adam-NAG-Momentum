# -classification-using-gradient-descent-and-its-varient-Adam-NAG-Momentum-

The goal of this Project is to implement and use gradient descent (and its variants) with backpropagation for a classification task.This network will be trained and tested using the Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784
pixels) from the Fashion-MNIST dataset,the network will be trained to classify the image into 1 of 10 classes.wiht differnet different loss function and ativation function like-sigmoid,tanh.
Dataset:Fashion-MNIST  

this code naive and simple approach to classify fashion-MNIST dataset using gradinet descent and its vareints and comparision between thse variants.Like:-
1) Adam
2) NAG
3) Momentum etc.


I have use two loss function and show comparision between thses loss function which one is performing better. 
1) Squared error loss
2) Cross Entropy Loss 

acivation function 
1) Sigmoid
2) tanh

below is the command with the appropriate parameter.Or use can directly use run.sh file for training and generating prediction.
#$/bin/bash
python train.py --lr 0.075 --momentum 0.075 --num_hidden 3 --sizes 100,100,100 --activation sigmoid --loss ce --opt gd --batch_size 20 --anneal true --save_dir pa1/ --expt_dir pa1/exp1/ --train train.csv --test test.csv --val val.csv --pretrain false
