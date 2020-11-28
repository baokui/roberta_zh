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