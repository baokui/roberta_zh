File=Docs

gpu=4
tag=allScenePre
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_allScene/ckpt/model.ckpt-200000'
bert_config_file='model/bert_allScene/bert_config.json'
vocab_file='model/bert_allScene/vocab.txt'
max_seqlen=32
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag >> log/sent-$tag-$File.log 2>&1 &

gpu=5
tag=allScenePre64
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_allScene64/ckpt/model.ckpt-200000'
bert_config_file='model/bert_allScene64/bert_config.json'
vocab_file='model/bert_allScene64/vocab.txt'
max_seqlen=64
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag >> log/sent-$tag-$File.log 2>&1 &

gpu=6
tag=allScenePre64-2
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_allScene64-2/ckpt/model.ckpt-1212000'
bert_config_file='model/bert_allScene64-2/bert_config.json'
vocab_file='model/bert_allScene64-2/vocab.txt'
max_seqlen=64
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag >> log/sent-$tag-$File.log 2>&1 &

gpu=2
tag=roberta24
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/roeberta_zh_L-24_H-1024_A-16/roberta_zh_large_model.ckpt'
bert_config_file='model/roeberta_zh_L-24_H-1024_A-16/bert_config_large.json'
vocab_file='model/roeberta_zh_L-24_H-1024_A-16/vocab.txt'
max_seqlen=128
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag >> log/sent-$tag-$File.log 2>&1 &


File=Docs
gpu=3
tag=allScenePre
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag-mean.json"
init_checkpoint='model/bert_allScene/ckpt/model.ckpt-200000'
bert_config_file='model/bert_allScene/bert_config.json'
vocab_file='model/bert_allScene/vocab.txt'
max_seqlen=32
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag >> log/sent-$tag-$File.log 2>&1 &

File=Queries
gpu=4
tag=allScenePre
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag-mean.json"
init_checkpoint='model/bert_allScene/ckpt/model.ckpt-200000'
bert_config_file='model/bert_allScene/bert_config.json'
vocab_file='model/bert_allScene/vocab.txt'
max_seqlen=32
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag >> log/sent-$tag-$File.log 2>&1 &

File=Queries1
gpu=4
tag=allScenePre-mean
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_allScene/ckpt/model.ckpt-200000'
bert_config_file='model/bert_allScene/bert_config.json'
vocab_file='model/bert_allScene/vocab.txt'
max_seqlen=32
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag >> log/sent-$tag-$File.log 2>&1 &


################################
# w2v weighted mean
File=Docs
gpu=6
tag=allScenePre-weightedmean
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_allScene/ckpt/model.ckpt-200000'
bert_config_file='model/bert_allScene/bert_config.json'
vocab_file='model/bert_allScene/vocab.txt'
path_idf='data_allScene_pretrain/IDF.json'
max_seqlen=32
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/sent-$tag-$File.log 2>&1 &

File=Queries1
gpu=6
tag=allScenePre-weightedmean
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_allScene/ckpt/model.ckpt-200000'
bert_config_file='model/bert_allScene/bert_config.json'
vocab_file='model/bert_allScene/vocab.txt'
path_idf='data_allScene_pretrain/IDF.json'
max_seqlen=32
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/sent-$tag-$File.log 2>&1 &


File=Docs
gpu=0
tag=allScenePre48-weightedmean
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_prose_finetune_mgpu/ckpt/model.ckpt'
bert_config_file='model/bert_prose_finetune_mgpu/bert_config.json'
vocab_file='model/bert_prose_finetune_mgpu/vocab.txt'
path_idf='data_allScene_pretrain/IDF.json'
max_seqlen=48
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/sent-$tag-$File.log 2>&1 &

File=Queries1
gpu=0
tag=allScenePre48-weightedmean
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_prose_finetune_mgpu/ckpt/model.ckpt'
bert_config_file='model/bert_prose_finetune_mgpu/bert_config.json'
vocab_file='model/bert_prose_finetune_mgpu/vocab.txt'
path_idf='data_allScene_pretrain/IDF.json'
max_seqlen=48
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/sent-$tag-$File.log 2>&1 &

File=Queries1
gpu=0
tag=allScenePre48-weightedmean-0
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_allScene48/ckpt/model.ckpt'
bert_config_file='model/bert_allScene48/bert_config.json'
vocab_file='model/bert_allScene48_0/vocab.txt'
path_idf='data_allScene_pretrain/IDF.json'
max_seqlen=48
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/sent-$tag-$File.log 2>&1 &

File=Docs
gpu=7
tag=allScenePre48-weightedmean-0
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_allScene48/ckpt/model.ckpt'
bert_config_file='model/bert_allScene48/bert_config.json'
vocab_file='model/bert_allScene48_0/vocab.txt'
path_idf='data_allScene_pretrain/IDF.json'
max_seqlen=48
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/sent-$tag-$File.log 2>&1 &

File=Docs
gpu=6
tag=allScenePre48-s2v
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
nohup python -u sentEmb_s2v.py $gpu $path_data $path_target $tag >> log/sent-$tag-$File.log 2>&1 &



File=Queries0121
gpu=3
tag=allScenePre48-weightedmean-finetune
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_allScene48/ckpt_finetune/model.ckpt'
bert_config_file='model/bert_allScene48/bert_config.json'
vocab_file='model/bert_allScene48/vocab.txt'
path_idf='data_allScene_pretrain/IDF.json'
max_seqlen=48
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/sent-$tag-$File.log 2>&1 &


File=Docs
gpu=0
tag=allScenePre48-weightedmean-flow
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/$File-$tag.json"
init_checkpoint='model/bert_allScene48/ckpt_flow'
bert_config_file='model/bert_allScene48/bert_config.json'
vocab_file='model/bert_allScene48/vocab.txt'
path_idf='data_allScene_pretrain/IDF.json'
max_seqlen=48
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/sent-$tag-$File.log 2>&1 &

########AI writer encoding############
gpu=3
tag=allScenePre48-weightedmean-finetune
path_data="/search/odin/guobk/vpa/AiWriter/DataAll/D_s2v_bert.json"
path_target="/search/odin/guobk/vpa/AiWriter/DataAll/D_s2v_bert.json"
init_checkpoint='model/bert_allScene48/ckpt_finetune/model.ckpt'
bert_config_file='model/bert_allScene48/bert_config.json'
vocab_file='model/bert_allScene48/vocab.txt'
path_idf='data_allScene_pretrain/IDF.json'
max_seqlen=48
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/sent-$tag-aiwrite.log 2>&1 &

gpu=3
tag="allScenePre48-weightedmean-finetune"
path_data="/search/odin/guobk/vpa/AiWriter/DataAll/D_s2v_bert.json"
path_target="/search/odin/guobk/vpa/AiWriter/DataAll/D_s2v_bert.json"
init_checkpoint='Data_AiWriter/model1/ckpt/model.ckpt-159000'
bert_config_file="Data_AiWriter/model1/bert_config.json"
vocab_file='model/bert_allScene48/vocab.txt'
path_idf='data_allScene_pretrain/IDF.json'
max_seqlen=128
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/sent-$tag-aiwrite.log 2>&1 &


gpu=3
tag=allScenePre48-weightedmean-finetune
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/test_s2v/Queries023_bertpre.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/test_s2v/Queries023_bertpre.json"
init_checkpoint='model/bert_allScene48/ckpt_finetune/model.ckpt'
bert_config_file='model/bert_allScene48/bert_config.json'
vocab_file='model/bert_allScene48/vocab.txt'
path_idf='data_allScene_pretrain/IDF.json'
max_seqlen=48
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/sent-$tag-aiwrite.log 2>&1 &

####################
# AiWriter-new data
for((gpu=1;gpu<6;gpu++))
do
  echo $gpu
tag="ai_pretrain"
path_data="/search/odin/guobk/data/AiWriter/Content/data/raw1-washed-dedup7-s2v-$gpu.json"
path_target="/search/odin/guobk/data/AiWriter/Content/data/raw1-washed-dedup7-s2v-$gpu-vector.json"
init_checkpoint='/search/odin/guobk/data/AiWriter/Content/data/pretrain/ckpt/model.ckpt-147000'
bert_config_file='/search/odin/guobk/data/AiWriter/Content/data/pretrain/bert_config.json'
vocab_file='/search/odin/guobk/data/AiWriter/Content/data/pretrain/vocab.txt'
path_idf='/search/odin/guobk/data/AiWriter/Content/data/IDF_char.json'
max_seqlen=128
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/aiwriteSentEmb-$gpu.log 2>&1 &
done

tag="ai_pretrain"
gpu=7
path_data="/search/odin/guobk/data/AiWriter/Content/data/sent2vec/test.json"
path_target="/search/odin/guobk/data/AiWriter/Content/data/sent2vec/test.json"
init_checkpoint='/search/odin/guobk/data/AiWriter/Content/data/pretrain/ckpt/model.ckpt-147000'
bert_config_file='/search/odin/guobk/data/AiWriter/Content/data/pretrain/bert_config.json'
vocab_file='/search/odin/guobk/data/AiWriter/Content/data/pretrain/vocab.txt'
path_idf='/search/odin/guobk/data/AiWriter/Content/data/IDF_char.json'
max_seqlen=128
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag $path_idf >> log/aiwriteSentEmb-test.log 2>&1 &
