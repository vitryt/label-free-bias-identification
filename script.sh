declare -a random_seeds=( 23 45 67 82 12 78 128 489 11 629)
declare -a number_of_concepts=( 10 30 70 100 300 500 700)
model="MNIST"
gpu_id="1"
result_path="/data/4vitry/"
data_path="/data/4vitry/"

for model_id in {5..10}
do
    echo "Starting experiment with model : $model_id"
    python model_training.py --model_id $model_id --model_name $model --epochs 40 --split_seed ${random_seeds[$model_id]} --shuffle_seed ${random_seeds[$model_id]} --gpu_id $gpu_id --result_path $result_path --data_path $data_path
    for concept_id in {0..6}
    do
        echo "Starting experiment : $model_id | $concept_id"
        python bias_identifying.py --model_id $model_id --model_name $model --concept_id $concept_id --number_of_concept ${number_of_concepts[$concept_id]} --gpu_id $gpu_id --result_path $result_path --data_path $data_path --concept_dataset_size 3000
        echo "Analysing biases : $model_id | $concept_id"
        python bias_analysis.py --model_id $model_id --model_name $model --concept_id $concept_id --gpu_id $gpu_id --result_path $result_path --data_path $data_path
    done
done
