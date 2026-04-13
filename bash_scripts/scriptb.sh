# Script for training the biased CMNIST models

declare -a random_seeds=( 23 45 67 82 12 78 128 489 11 629 73 198 74 26 50 52 958 91 18 4 65 99 27 84 39 56)
dataset="MNIST"
model_type="MLP"
model_name="CMNISTb"
optimizer="sgd"
gpu_id="0"
result_path=""
data_path="data"

for model_id in {0..9}
do
    echo "Starting experiment with model : $model_id"
    python model_training.py --model_id $model_id --model_name $model_name --batch_size 128 --epochs 100 --split_seed ${random_seeds[$model_id]} --shuffle_seed ${random_seeds[$model_id]} --gpu_id $gpu_id --result_path $result_path --data_path $data_path --dataset $dataset --model_type $model_type --optimizer $optimizer --train_correlation "0.95" --test_correlation "0.1"
done