## lv

python train_lv.py --model_dir=models/lv/lv_4hg_lr1e-3_decay10 --n_hourglass=4 --train_data=data/tfrecords/lv/train.tfrecords --eval_data=data/tfrecords/lv/test.tfrecords  --initial_learning_rate=0.01 




## CATPOSE
```
python train_catpose.py \
--model_dir=models/catpose/catpose_2hg_lr1e-3_decay10 \
--n_hourglass=2 \
--train_data=data/tfrecords/catpose/train.tfrecords \
--eval_data=data/tfrecords/catpose/eval.tfrecords \
--eval_every_n_epochs=1 \
--patience=10 \
--initial_learning_rate=0.001 \
--learning_rate_decay_epochs=10
```
