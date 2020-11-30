path_data='tianchi/data/prepro/dev.txt'
path_target='tianchi/data/prepro/dev-30000'
path_model='tianchi/model/model.ckpt-30000'
nohup python -u test_tianchi.py $path_data $path_target $path_model >> log/test_tianchi-30000.log 2>&1 &

path_data='tianchi/data/prepro/dev.txt'
path_target='tianchi/data/prepro/dev-30000'
path_model='tianchi/model128/model.ckpt-30000'
max_seqlen=128
nohup python -u test_tianchi.py $path_data $path_target $path_model $max_seqlen >> log/test_tianchi128-3000.log 2>&1 &