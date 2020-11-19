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
