export CUDA_VISIBLE_DEVICES="0"
nohup python -u finetune_s2v.py >> log/finetune_s2v.log 2>&1 &
export CUDA_VISIBLE_DEVICES="7"
nohup python -u finetune_s2v.py \
     --output_dir=model/model_s2v/ckpt2 >> log/finetune_s2v-2.log 2>&1 &