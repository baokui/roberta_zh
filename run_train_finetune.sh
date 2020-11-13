BERT_BASE_DIR=$1
cuda=$2
my_new_model_path=$3
export CUDA_VISIBLE_DEVICES=$cuda
nohup python -u run_pretraining.py --input_file=data_prose/raw5-text.tfrecord  \
--output_dir=$my_new_model_path --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json \
--train_batch_size=64 --max_seq_length=256 --max_predictions_per_seq=23 \
--num_train_steps=200000 --num_warmup_steps=10000 --learning_rate=1e-4  --tpu_name=123  \
--save_checkpoints_steps=3000  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt >> log/finetune-$BERT_BASE_DIR.log 2>&1 &