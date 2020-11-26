gpu=4
tag=allScenePre
path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/Docs.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/Docs-$tag.json"
init_checkpoint='model/bert_allScene/ckpt/model.ckpt-200000'
bert_config_file='model/bert_allScene/bert_config.json'
vocab_file='model/bert_allScene/vocab.txt'
max_seqlen=32
nohup python -u sentEmb.py $gpu $path_data $path_target $init_checkpoint $bert_config_file $vocab_file $max_seqlen $tag >> log/sent-$tag.log 2>&1 &