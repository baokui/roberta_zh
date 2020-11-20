BERT_BASE_DIR=model/bert_allScene
my_new_model_path=$BERT_BASE_DIR/ckpt
mkdir -p $my_new_model_path
max_seq_length=32
export CUDA_VISIBLE_DEVICES=3
filelist=`ls data_allScene_pretrain/tfrecord/`
array=($filelist)
files=data_allScene_pretrain/tfrecord/${array[0]}
for((i=1;i<${#array[@]};i++))
do
  files=$files,data_allScene_pretrain/tfrecord/${array[i]}
done

nohup python -u run_pretraining.py --input_file=$files  \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=64 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=200000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000  >> log/pretrain-bert_allScene.log 2>&1 &

##############################3

BERT_BASE_DIR=model/bert_allScene64
my_new_model_path=$BERT_BASE_DIR/ckpt
mkdir -p $my_new_model_path
max_seq_length=64
export CUDA_VISIBLE_DEVICES=3
filelist=`ls data_allScene_pretrain/tfrecord64/`
array=($filelist)
files=data_allScene_pretrain/tfrecord64/${array[0]}
for((i=1;i<${#array[@]};i++))
do
  b=$(( $i % 7 ))
  if [ $b == 0 ]; then
    files=$files,data_prose/tfrecord64/raw.tfrecord
    echo aaa
  fi
  files=$files,data_allScene_pretrain/tfrecord64/${array[i]}
done
echo $files

nohup python -u run_pretraining.py --input_file=$files  \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=64 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=200000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000  >> log/pretrain-bert_allScene64.log 2>&1 &

