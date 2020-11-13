modelfile=model/roberta_zh_l12
cuda=2
outputfile=model/roberta-12-finetune
max_seq_length=1024
mkdir $outputfile
sh run_train_finetune.sh $modelfile $cuda $outputfile $max_seq_length

modelfile=model/roberta_zh_L-6-H-768_A-12
cuda=3
outputfile=model/roberta-6-finetune
max_seq_length=256
mkdir $outputfile
sh run_train_finetune.sh $modelfile $cuda $outputfile $max_seq_length

modelfile=model/roeberta_zh_L-24_H-1024_A-16
cuda=4
outputfile=model/roberta-24-finetune
max_seq_length=1024
mkdir $outputfile
sh run_train_finetune.sh $modelfile $cuda $outputfile $max_seq_length