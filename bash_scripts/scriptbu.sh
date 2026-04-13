# Script for identifying the bias in the biased CMNIST models using unbiased audit
declare -a random_seeds=( 23 45 67 82 12 78 128 489 11 629 73 198 74 26 50 52 958 91 18 4 65 99 27 84 39)
declare -a number_of_concepts=( 6 8 10 12 )
declare -a patch_sizes=( 4 6 8 10 )
experiment_name="u"
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
    for number_of_concept in "${number_of_concepts[@]}"
    do
        for patch_size in "${patch_sizes[@]}"
        do
            experiment_tag="${experiment_name}_${number_of_concept}_${patch_size}"
            echo "Starting experiment : $model_id | experiment_tag=$experiment_tag"
            python bias_identifying.py --model_id $model_id --model_name $model_name --experiment_name $experiment_name --number_of_concept $number_of_concept  --patch_size $patch_size --gpu_id $gpu_id --result_path $result_path --data_path $data_path --val_correlation 0.1
            echo "Analysing biases : $model_id | experiment_tag=$experiment_tag"
            python bias_analysis.py --model_id $model_id --model_name $model_name --experiment_name $experiment_name --number_of_concept $number_of_concept --patch_size $patch_size --gpu_id $gpu_id --result_path $result_path --data_path $data_path
        done
    done
    python debiasing_expe.py --model_id $model_id --model_name $model_name --experiment_name $experiment_name --number_of_concept 8 --patch_size 6 --gpu_id $gpu_id --result_path $result_path --data_path $data_path --concept_threshold "0.55" --backprop_step 20000 
done