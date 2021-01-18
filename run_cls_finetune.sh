export CUDA_VISIBLE_DEVICES=2
BERT_BASE_DIR=model/roberta_zh_l12
task_name=使用场景P0
mkdir -p model/label/$task_name
nohup python -u run_cls_finetune.py \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --task_name=$task_name \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --output_dir=model/label/$task_name \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=512 \
    --do_train=True \
    --do_eval=True \
    >> log/labelmodel-$task_name.log 2>&1 &