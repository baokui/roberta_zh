export CUDA_VISIBLE_DEVICES=2
nohup python -u run_cls_finetune.py >> log/labelmodel.log 2>&1 &