BERT_BASE_DIR=model/roeberta_zh_L-24_H-1024_A-16
nohup python -u run_classifier_tianchi.py \
--data_dir=tianchi/data/prepro/  \
--output_dir=tianchi/model/ \
--do_train=True \
--do_eval=True \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/roberta_zh_large_model.ckpt \
--task_name=lcqmc_pair \
--train_batch_size=32 \
--max_seq_length=256 \
--max_predictions_per_seq=2 \
--num_train_steps=20000000 \
--num_warmup_steps=10000 \
--learning_rate=1e-4  \
--tpu_name=123  \
--save_checkpoints_steps=3000  >> log/tianchi-roberta-24.log 2>&1 &