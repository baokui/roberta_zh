export CUDA_VISIBLE_DEVICES="0"
nohup python -u finetune_s2v.py >> log/finetune_s2v.log 2>&1 &