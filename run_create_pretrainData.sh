inputfile="data_prose/raw5-text.txt"
outputfile="data_prose/raw5-text.tfrecord"
python create_pretraining_data.py --do_whole_word_mask=True --input_file=$inputfile \
--output_file=$outputfile --vocab_file=./resources/vocab.txt \
--do_lower_case=True --max_seq_length=256 --max_predictions_per_seq=23 --masked_lm_prob=0.15  --random_seed=12345  --dupe_factor=5 \
>> log/create-pretrain.log 2>&1 &
