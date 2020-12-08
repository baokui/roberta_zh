path_data="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/Docs-allScenePre.json"
path_target="/search/odin/guobk/vpa/vpa-studio-research/retrieval/data/Docs-allScenePre-bert3.json"
nohup python -u tfserving_bert.py $path_data $path_target test >> log/sentemb-bert3.log 2 >&1 &
