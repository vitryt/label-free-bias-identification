declare -a random_seeds=( 23 45 67 82 12 78 128 489 11 629 73 198 74 26 50 52 958 91 18 4 65 99 27 84 39)
declare -a number_of_concepts=( 4 8 16 32 )
declare -a patch_sizes=( 4 8 16 28 )
model="MNIST"
gpu_id="1"
result_path="/data/4vitry/"
data_path="/data/4vitry/"

for model_id in {10..15}
do
    echo "Starting experiment with model : $model_id"
    python model_testing.py --model_id $model_id --model_name $model --gpu_id $gpu_id --result_path $result_path --data_path $data_path
    # for concept_id in {0..4}
    # do
        # for patch_size_id in {0..4}
        # do
            # echo "Starting experiment : $model_id | $concept_id, $patch_size_id"
            # python bias_identifying.py --model_id $model_id --model_name $model --concept_id $concept_id"_"$patch_size_id --number_of_concept ${number_of_concepts[$concept_id]} --gpu_id $gpu_id --result_path $result_path --data_path $data_path --concept_dataset_size 1000 --multi_concept 1 --patch_size ${patch_sizes[$patch_size_id]}
            # echo "Analysing biases : $model_id | $concept_id, $patch_size_id"
            # python bias_analysis.py --model_id $model_id --model_name $model --concept_id $concept_id"_"$patch_size_id --gpu_id $gpu_id --result_path $result_path --data_path $data_path --multi_concept 1
            # echo "Correlating biases : $model_id | $concept_id, $patch_size_id"
            # python bias_correlation.py --model_id $model_id --model_name $model --concept_id $concept_id"_"$patch_size_id --gpu_id $gpu_id --result_path $result_path --data_path $data_path --multi_concept 1
        # done
    # done
done
