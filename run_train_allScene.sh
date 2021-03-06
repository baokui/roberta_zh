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
export CUDA_VISIBLE_DEVICES=7
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
--num_train_steps=2000000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000  >> log/pretrain-bert_allScene64.log 2>&1 &



BERT_BASE_DIR=model/bert_allScene64-2
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
nohup python -u run_pretraining.py --input_file=$files  \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=64 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=2000000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 --init_checkpoint=model/bert_allScene64/ckpt/model.ckpt-200000 >> log/pretrain-bert_allScene64-2.log 2>&1 &


BERT_BASE_DIR=model/bert_prose
my_new_model_path=$BERT_BASE_DIR/ckpt
mkdir -p $my_new_model_path
max_seq_length=48
export CUDA_VISIBLE_DEVICES=3
files=data_prose/tfrecord48/raw.tfrecord
nohup python -u run_pretraining.py --input_file=$files  \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=64 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=200000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 >> log/pretrain-bert_prose48.log 2>&1 &

BERT_BASE_DIR=model/bert_prose_finetune
my_new_model_path=$BERT_BASE_DIR/ckpt
mkdir -p $my_new_model_path
max_seq_length=48
export CUDA_VISIBLE_DEVICES=3
filelist=`ls data_allScene_pretrain/tfrecord48/`
array=($filelist)
files=data_allScene_pretrain/tfrecord48/${array[0]}
for((i=1;i<${#array[@]};i++))
do
  files=$files,data_allScene_pretrain/tfrecord48/${array[i]}
done
nohup python -u run_pretraining.py --input_file=$files  \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=64 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=2000000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 --init_checkpoint=model/bert_prose_finetune/ckpt/model.ckpt-342000 >> log/pretrain-bert_prose_finetune48-2.log 2>&1 &



BERT_BASE_DIR=model/bert_prose_finetune_mgpu
my_new_model_path=$BERT_BASE_DIR/ckpt
mkdir -p $my_new_model_path
max_seq_length=48
export CUDA_VISIBLE_DEVICES="1,2,4,5"
filelist=`ls data_allScene_pretrain/tfrecord48/`
array=($filelist)
files=data_allScene_pretrain/tfrecord48/${array[0]}
for((i=1;i<${#array[@]};i++))
do
  files=$files,data_allScene_pretrain/tfrecord48/${array[i]}
done
nohup python -u run_pretraining_mGPU.py --input_file=$files  --n_gpus=4 \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=64 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=2000000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 --init_checkpoint=model/bert_prose_finetune_mgpu/ckpt/model.ckpt >> log/pretrain-bert_prose_finetune48-gpus-2.log 2>&1 &


BERT_BASE_DIR=model/bert_allScene48
my_new_model_path=$BERT_BASE_DIR/ckpt
mkdir -p $my_new_model_path
max_seq_length=48
export CUDA_VISIBLE_DEVICES="1,2,4,5"
filelist=`ls data_allScene_pretrain/tfrecord48/`
array=($filelist)
files=data_allScene_pretrain/tfrecord48/${array[0]}
for((i=1;i<${#array[@]};i++))
do
  files=$files,data_allScene_pretrain/tfrecord48/${array[i]}
done
nohup python -u run_pretraining_mGPU.py --input_file=$files  --n_gpus=4 \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=64 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=2000000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 >> log/pretrain-bert_allScene48.log 2>&1 &

BERT_BASE_DIR=model/bert_allScene48_0
my_new_model_path=$BERT_BASE_DIR/ckpt1
mkdir -p $my_new_model_path
max_seq_length=48
export CUDA_VISIBLE_DEVICES=0
filelist=`ls data_allScene_pretrain/tfrecord48/`
array=($filelist)
files=data_allScene_pretrain/tfrecord48/${array[0]}
for((i=1;i<${#array[@]};i++))
do
  files=$files,data_allScene_pretrain/tfrecord48/${array[i]}
done

nohup python -u run_pretraining.py --input_file=$files  \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=64 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=200000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 --init_checkpoint=model/bert_allScene48_0/ckpt/model.ckpt-78000  >> log/pretrain-bert_allScene48-0-1.log 2>&1 &

BERT_BASE_DIR=model/bert_aiwriter
my_new_model_path=$BERT_BASE_DIR/ckpt
datapath=data_aiwriter/tfrecord
mkdir -p $my_new_model_path
max_seq_length=128
export CUDA_VISIBLE_DEVICES=6
filelist=`ls $datapath`
array=($filelist)
files=$datapath/${array[0]}
for((i=1;i<${#array[@]};i++))
do
  files=$files,$datapath/${array[i]}
done
echo $files,$my_new_model_path,$BERT_BASE_DIR/bert_config.json,$max_seq_length
nohup python -u run_pretraining.py \
  --input_file=$files  \
  --output_dir=$my_new_model_path \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=64 \
  --max_seq_length=$max_seq_length \
  --max_predictions_per_seq=2 \
  --num_train_steps=200000 \
  --num_warmup_steps=10000 \
  --learning_rate=1e-4  \
  --tpu_name=123  \
  --save_checkpoints_steps=3000  >> log/pretrain-bert_aiwriter.log 2>&1 &


BERT_BASE_DIR=model/bert_allScene48
my_new_model_path=$BERT_BASE_DIR/ckpt_finetune
init_checkpoint=$BERT_BASE_DIR/ckpt/model.ckpt
mkdir -p $my_new_model_path
max_seq_length=48
export CUDA_VISIBLE_DEVICES="2,3,4,5"
files=data_prose/tfrecord48/raw.tfrecord
nohup python -u run_pretraining_mGPU.py --input_file=$files  --n_gpus=4 \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=64 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=20000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 --init_checkpoint=$init_checkpoint >> log/pretrain-bert_allScene48-finetune.log 2>&1 &


BERT_BASE_DIR=model/bert_allScene48
my_new_model_path=$BERT_BASE_DIR/ckpt_flow
init_checkpoint=$BERT_BASE_DIR/ckpt/model.ckpt
mkdir -p $my_new_model_path
max_seq_length=48
export CUDA_VISIBLE_DEVICES="7"
filelist=`ls data_allScene_pretrain/tfrecord48/`
array=($filelist)
files=data_allScene_pretrain/tfrecord48/${array[0]}
for((i=1;i<${#array[@]};i++))
do
  files=$files,data_allScene_pretrain/tfrecord48/${array[i]}
done
nohup python -u run_pretraining_mGPU_flow.py --input_file=$files  --n_gpus=1 \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=64 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=2000000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 --init_checkpoint=$init_checkpoint >> log/pretrain-bert_allScene48-flow.log 2>&1 &




my_new_model_path=Data_AiWriter/model/ckpt
bert_config_file=Data_AiWriter/model/bert_config.json
init_checkpoint=model/roberta_zh_L-6-H-768_A-12/bert_model.ckpt
mkdir -p $my_new_model_path
max_seq_length=512
export CUDA_VISIBLE_DEVICES="4"
files=Data_AiWriter/data/tfrecord/raw.tfrecord
nohup python -u run_pretraining.py --input_file=$files  --n_gpus=1 \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$bert_config_file \
--train_batch_size=64 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=2000000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 --init_checkpoint=$init_checkpoint >> log/pretrain-AiWriter.log 2>&1 &

BERT_BASE_DIR=Data_AiWriter/model/
my_new_model_path=Data_AiWriter/model/ckpt
init_checkpoint=model/roberta_zh_L-6-H-768_A-12/bert_model.ckpt
mkdir -p $my_new_model_path
max_seq_length=512
export CUDA_VISIBLE_DEVICES="2,3,4,5"
files=Data_AiWriter/data/tfrecord/raw.tfrecord
nohup python -u run_pretraining_mGPU.py --input_file=$files  --n_gpus=4 \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=16 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=20000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 --init_checkpoint=$init_checkpoint >> log/pretrain-AiWriter.log 2>&1 &


BERT_BASE_DIR=Data_AiWriter/model1/
my_new_model_path=Data_AiWriter/model1/ckpt
init_checkpoint=model/roberta_zh_L-6-H-768_A-12/bert_model.ckpt
mkdir -p $my_new_model_path
max_seq_length=128
export CUDA_VISIBLE_DEVICES="2,3,4,5"
files=Data_AiWriter/data/tfrecord/raw.tfrecord
nohup python -u run_pretraining_mGPU.py --input_file=$files  --n_gpus=4 \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=16 --max_seq_length=$max_seq_length --max_predictions_per_seq=2 \
--num_train_steps=100000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 --init_checkpoint=$init_checkpoint >> log/pretrain-AiWriter.log 2>&1 &

BERT_BASE_DIR=Data_AiWriter/model1/
my_new_model_path=Data_AiWriter/model1/ckpt
init_checkpoint=model/roberta_zh_L-6-H-768_A-12/bert_model.ckpt
mkdir -p $my_new_model_path
max_seq_length=128
export CUDA_VISIBLE_DEVICES="3"
files=Data_AiWriter/data1/tfrecord/raw.tfrecord
init_checkpoint=model/roberta_zh_L-6-H-768_A-12/bert_model.ckpt
nohup python -u run_pretraining.py --input_file=$files  \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=16 --max_seq_length=$max_seq_length --max_predictions_per_seq=4 \
--num_train_steps=2000000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000 --init_checkpoint=$init_checkpoint >> log/pretrain-AiWriter1.log 2>&1 &

#########################################################################3
# 标签模型预训练训练（句库提供）
BERT_BASE_DIR=/search/odin/guobk/data/labels/data_new/model_pretrain
my_new_model_path=$BERT_BASE_DIR/ckpt/
mkdir -p $my_new_model_path
max_seq_length=32
export CUDA_VISIBLE_DEVICES="0"
files=/search/odin/guobk/data/labels/data_new/pretrainData/tfrecord/raw.tfrecord
init_checkpoint=model/roberta_zh_L-6-H-768_A-12/bert_model.ckpt
nohup python -u run_pretraining.py \
  --input_file=$files  \
  --output_dir=$my_new_model_path \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=16 \
  --max_seq_length=$max_seq_length \
  --max_predictions_per_seq=4 \
  --num_train_steps=2000000 \
  --num_warmup_steps=10000 \
  --learning_rate=1e-4  \
  --tpu_name=123  \
  --save_checkpoints_steps=3000 \
  --init_checkpoint=$init_checkpoint >> log/pretrain-labels.log 2>&1 &
#########################################################################3
# 素材推荐-素材预训练
BERT_BASE_DIR=/search/odin/guobk/data/AiWriter/Content/data/pretrain/
my_new_model_path=$BERT_BASE_DIR/ckpt/
mkdir -p $my_new_model_path
max_seq_length=128
export CUDA_VISIBLE_DEVICES="0"
files=/search/odin/guobk/data/AiWriter/Content/data/pretrain/tfrecord/train.tfrecord
init_checkpoint=model/roberta_zh_L-6-H-768_A-12/bert_model.ckpt
nohup python -u run_pretraining.py \
  --input_file=$files  \
  --output_dir=$my_new_model_path \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=16 \
  --max_seq_length=$max_seq_length \
  --max_predictions_per_seq=4 \
  --num_train_steps=2000000 \
  --num_warmup_steps=10000 \
  --learning_rate=1e-4  \
  --tpu_name=123  \
  --save_checkpoints_steps=3000 \
  --init_checkpoint=$init_checkpoint >> log/pretrain-AiWriter-new.log 2>&1 &