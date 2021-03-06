inputfile="data_prose/raw5-text.txt"
outputfile="data_prose/raw5-text.tfrecord"
python -u create_pretraining_data.py --do_whole_word_mask=True --input_file=$inputfile \
--output_file=$outputfile --vocab_file=./resources/vocab.txt \
--do_lower_case=True --max_seq_length=256 --max_predictions_per_seq=23 --masked_lm_prob=0.15  --random_seed=12345  --dupe_factor=5 \
>> log/create-pretrain.log 2>&1 &


filelist=`ls data_allScene_pretrain/raw-washed/`
list1=($filelist)
for((i=30;i<40;i++))
do
file=${list1[i]}
 echo $file
 inputfile=data_allScene_pretrain/raw-washed/$file
 outputfile=data_allScene_pretrain/tfrecord/$file.tfrecord
 nohup python -u create_pretraining_data.py --do_whole_word_mask=True --input_file=$inputfile \
--output_file=$outputfile --vocab_file=./resources/vocab.txt \
--do_lower_case=True --max_seq_length=32 --max_predictions_per_seq=2 --masked_lm_prob=0.1  --random_seed=12345  --dupe_factor=1 \
>> log/create-pretrain-$file.log 2>&1 &
done


filelist=`ls data_allScene_pretrain/raw-washed/`
list1=($filelist)
for((i=20;i<30;i++))
do
file=${list1[i]}
 echo $file
 inputfile=data_allScene_pretrain/raw-washed/$file
 outputfile=data_allScene_pretrain/tfrecord64/$file.tfrecord
 nohup python -u create_pretraining_data.py --do_whole_word_mask=True --input_file=$inputfile \
--output_file=$outputfile --vocab_file=./resources/vocab.txt \
--do_lower_case=True --max_seq_length=64 --max_predictions_per_seq=2 --masked_lm_prob=0.1  --random_seed=12345  --dupe_factor=1 \
>> log/create-pretrain-64-$file.log 2>&1 &
done

inputfile=data_prose/raw.txt
outputfile=data_prose/tfrecord64/raw.tfrecord
 nohup python -u create_pretraining_data.py --do_whole_word_mask=True --input_file=$inputfile \
--output_file=$outputfile --vocab_file=./resources/vocab.txt \
--do_lower_case=True --max_seq_length=64 --max_predictions_per_seq=2 --masked_lm_prob=0.1  --random_seed=12345  --dupe_factor=1 \
>> log/create-pretrain-64-prose.log 2>&1 &

inputfile=data_prose/raw.txt
outputfile=data_prose/tfrecord48/raw.tfrecord
 nohup python -u create_pretraining_data.py --do_whole_word_mask=True --input_file=$inputfile \
--output_file=$outputfile --vocab_file=./resources/vocab.txt \
--do_lower_case=True --max_seq_length=48 --max_predictions_per_seq=2 --masked_lm_prob=0.1  --random_seed=12345  --dupe_factor=1 \
>> log/create-pretrain-48-prose.log 2>&1 &

filelist=`ls data_allScene_pretrain/raw-washed/`
list1=($filelist)
for((i=40;i<50;i++))
do
file=${list1[i]}
 echo $file
 inputfile=data_allScene_pretrain/raw-washed/$file
 outputfile=data_allScene_pretrain/tfrecord48/$file.tfrecord
 nohup python -u create_pretraining_data.py --do_whole_word_mask=True --input_file=$inputfile \
--output_file=$outputfile --vocab_file=./resources/vocab.txt \
--do_lower_case=True --max_seq_length=48 --max_predictions_per_seq=2 --masked_lm_prob=0.1  --random_seed=12345  --dupe_factor=1 \
>> log/create-pretrain-$file.log 2>&1 &
done


#!!!!!! inputfile里的内容必须是\n\n分割！！！！！！！！！！！！
mkdir data_aiwriter/tfrecord/
for((i=0;i<8;i++))
do
inputfile="data_aiwriter/washed1-$i.txt"
outputfile="data_aiwriter/tfrecord/data-$i.tfrecord"
nohup python -u create_pretraining_data.py \
  --do_whole_word_mask=True \
  --input_file=$inputfile \
  --output_file=$outputfile \
  --vocab_file=./resources/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --min_seg_length=20 \
  --max_predictions_per_seq=2 \
  --masked_lm_prob=0.1  \
  --random_seed=12345  \
  --dupe_factor=1 >> log/create-pretrain-data_aiwriter-$i.log 2>&1 &
done


#!!!!!! inputfile里的内容必须是\n\n分割！！！！！！！！！！！！
inputfile="Data_AiWriter/data/raw.txt"
outputfile="Data_AiWriter/data/tfrecord/raw.tfrecord"
nohup python -u create_pretraining_data.py \
  --do_whole_word_mask=True \
  --input_file=$inputfile \
  --output_file=$outputfile \
  --vocab_file=./resources/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=512 \
  --min_seg_length=10 \
  --max_predictions_per_seq=2 \
  --masked_lm_prob=0.1  \
  --random_seed=12345  \
  --dupe_factor=5 >> log/create-pretrain-Aiwriter.log 2>&1 &


#!!!!!! inputfile里的内容必须是\n\n分割！！！！！！！！！！！！
inputfile="Data_AiWriter/data1/raw.txt"
outputfile="Data_AiWriter/data1/tfrecord/raw.tfrecord"
nohup python -u create_pretraining_data.py \
  --do_whole_word_mask=True \
  --input_file=$inputfile \
  --output_file=$outputfile \
  --vocab_file=./resources/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --min_seg_length=10 \
  --max_predictions_per_seq=4 \
  --masked_lm_prob=0.1  \
  --random_seed=12345  \
  --dupe_factor=5 >> log/create-pretrain-Aiwriter1.log 2>&1 &

#!!!!!! inputfile里的内容必须是\n\n分割！！！！！！！！！！！！
inputfile="/search/odin/guobk/data/labels/data_new/pretrainData/raw.txt"
mkdir /search/odin/guobk/data/labels/data_new/pretrainData/tfrecord
outputfile="/search/odin/guobk/data/labels/data_new/pretrainData/tfrecord/raw.tfrecord"
nohup python -u create_pretraining_data.py \
  --do_whole_word_mask=True \
  --input_file=$inputfile \
  --output_file=$outputfile \
  --vocab_file=/search/odin/guobk/data/labels/data_new/model_pretrain/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=32 \
  --min_seg_length=10 \
  --max_predictions_per_seq=4 \
  --masked_lm_prob=0.1  \
  --random_seed=12345  \
  --dupe_factor=5 >> log/create-pretrain-labels-classify.log 2>&1 &

###########################################################
# 素材推荐预训练
#!!!!!! inputfile里的内容必须是\n\n分割！！！！！！！！！！！！
inputfile="/search/odin/guobk/data/AiWriter/Content/data/pretrain/train.txt"
mkdir /search/odin/guobk/data/AiWriter/Content/data/pretrain/tfrecord
outputfile=/search/odin/guobk/data/AiWriter/Content/data/pretrain/tfrecord/train.tfrecord
nohup python -u create_pretraining_data.py \
  --do_whole_word_mask=True \
  --input_file=$inputfile \
  --output_file=$outputfile \
  --vocab_file=/search/odin/guobk/data/AiWriter/Content/data/pretrain/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --min_seg_length=10 \
  --max_predictions_per_seq=4 \
  --masked_lm_prob=0.1  \
  --random_seed=12345  \
  --dupe_factor=5 >> log/create-pretrain-aiwriter-new.log 2>&1 &