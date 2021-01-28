export CUDA_VISIBLE_DEVICES="2"
path_config="config/aiwriter.config.json"
port=1004
ps -ef|grep $port|grep server_bertEmbedding.py|grep -v grep|awk  '{print "kill -9 " $2}' |sh
sleep 1s
nohup python -u server_bertEmbedding.py $path_config $port >> log/server-bertEmb-$port.log 2>&1 &