declare -a random_seeds=( 23 45 67 82 12 78 128 489 11 629 73 198 74 26 50 52 958 91 18 4 65 99 27 84 39 56)
declare -a number_of_concepts=( 10 15 20 )
declare -a patch_sizes=( 25 50 75 )
dataset="Waterbirds"
model_type="resnet18"
model_name="Waterbirds18"
experiment_name="b"
optimizer="sgd"
gpu_id="0"
result_path=""
data_path="data"

for model_id in {0..9}
do
    echo "Starting experiment with model : $model_id"
    python model_training.py --model_id $model_id --model_name $model_name --batch_size 512 --epochs 100 --split_seed ${random_seeds[$model_id]} --shuffle_seed ${random_seeds[$model_id]} --gpu_id $gpu_id --result_path $result_path --data_path $data_path --dataset $dataset --model_type $model_type --optimizer $optimizer --lr_decrease_factor 0.1 --lr_decrease_frequency 30
    for number_of_concept in "${number_of_concepts[@]}"
    do
        for patch_size in "${patch_sizes[@]}"
        do
            experiment_tag="${experiment_name}_${number_of_concept}_${patch_size}"
            echo "Starting experiment : $model_id | experiment_tag=$experiment_tag"
            python bias_identifying.py --model_id $model_id --model_name $model_name --experiment_name $experiment_name --number_of_concept $number_of_concept  --patch_size $patch_size --gpu_id $gpu_id --result_path $result_path --data_path $data_path
        done
    done
    python debiasing_expe.py --model_id $model_id --model_name $model_name --experiment_name $experiment_name --number_of_concept 10 --patch_size 50 --gpu_id $gpu_id --result_path $result_path --data_path $data_path --concept_threshold "0.55" --backprop_step 20000 --overwrite_phase 6
done