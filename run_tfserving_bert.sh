source=/search/odin/guobk/vpa/roberta_zh/model/bert_prose_finetune_mgpu/pbmodel/
model=bert3
target=/models/$model
docker run -p 8503:8501 --mount type=bind,source=$source,target=$target -e MODEL_NAME=$model -t tensorflow/serving >> ./log/tfserving-cpu-$model.log 2>&1 &
curl http://localhost:8503/v1/models/$model/versions/1
curl http://localhost:8503/v1/models/$model #查看模型所有版本服务状态
curl http://localhost:8503/v1/models/$model/metadata #查看服务信息，输入大小等

url=http://localhost:8503/v1/models/$model:predict

input_ids='[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]'
input_mask='[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]'
segment_ids='[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]'
#data='{"instances": '$input_ids'}'
#curl -d $data -X POST $url

curl -d '{"instances": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]}' -X POST $url