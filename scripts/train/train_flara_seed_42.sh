
SEED=42
LEARNING_RATE=2.5e-5

timestamp=$(date +"%Y%m%d_%H%M%S")
experiment_name="train_flara_nexar_seed${SEED}_${timestamp}"
base_path="<output_root>"
output_path="${base_path}/${experiment_name}"
mkdir -p "$output_path"
echo "Created output directory: $output_path"
log_dir="${output_path}/logs"
mkdir -p "$log_dir"
echo "Created log directory: $log_dir"
log_file="${log_dir}/output_log.log"
error_file="${log_dir}/error_log.log"

echo "Logging to: $log_file"

python -u train.py \
    --train_csv <path_to_nexar_metadata.csv> \
    --val_csv "" \
    --test_csv <path_to_nexar_metadata.csv> \
    --data_root <nexar_data_root> \
    --label_mapping_path dataset/nexar_label_mapping.json \
    --dataset_name nexar \
    --num_frames 16 \
    --frame_size 256 \
    --batch_size 4 \
    --num_workers 8 \
    --use_fixed_fps False \
    --learning_rate $LEARNING_RATE \
    --epochs 30 \
    --model_name "facebook/vjepa2-vitl-fpc16-256-ssv2" \
    --trainable_parameters_configuration last_block+predictor+pool+head \
    --experiment_name "$experiment_name" \
    --output_path "$base_path" \
    --gpu_id 0 \
    --log_path "$log_dir" \
    --lambda_cls 1.0 \
    --lambda_prediction 10.0 \
    --num_classes 2 \
    --duration 2.0 \
    --fps 8 \
    --seed $SEED \
    --anticipation_offset_min 0.0 \
    --anticipation_offset_max 1.5 \
    --use_time_of_alert_offset True \
    --pooling_mode "attentive" \
    --predict_future_temporal_steps 8 \
    --prediction_future_frames 16 \
    --classify_on_predicted_only True \
    1> >(tee "$log_file") \
    2> >(tee "$error_file" >&2)


#### START TESTING SLIDING WINDOW SCRIPTS ####

MODEL_PATH="${output_path}/best_model.pth"
NAME_EXP="${experiment_name}"

base_output_dir="<eval_output_root>"

NEXAR_CSV="<path_to_nexar_metadata.csv>"
NEXAR_DATA_ROOT="<nexar_data_root>"

DAD_CSV="<path_to_dad_test.csv>"
DAD_DATA_ROOT="<dad_data_root>"

DADA2000_CSV="<path_to_dada2000_test.csv>"
DADA2000_DATA_ROOT="<dada2000_data_root>"

DOTA_CSV="<path_to_dota_test.csv>"
DOTA_DATA_ROOT="<dota_data_root>"

MODEL_NAME="facebook/vjepa2-vitl-fpc16-256-ssv2"
POOLING_MODE="attentive"
PREDICT_FUTURE=8

NUM_FRAMES=16
FPS=8.0
FRAME_SIZE=256
NUM_CLASSES=2
SLIDING_WINDOW_STRIDE=8

timestamp=$(date +"%Y%m%d_%H%M%S")
experiment_name="sliding_window_${NAME_EXP}_stride${SLIDING_WINDOW_STRIDE}_${timestamp}"
output_dir="${base_output_dir}/${experiment_name}"
mkdir -p "$output_dir"
log_file="${output_dir}/output_log.log"
error_file="${output_dir}/error_log.log"

echo "Running VJEPA2 sliding window evaluation on all datasets"
echo "Output dir: $output_dir"
echo "Logging to: $log_file"

python -u evaluate_sliding_window.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --pooling_mode "$POOLING_MODE" \
    --datasets nexar dad dada2000 dota \
    --num_classes "$NUM_CLASSES" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --frame_size "$FRAME_SIZE" \
    --sliding_window_stride "$SLIDING_WINDOW_STRIDE" \
    --nexar_csv "$NEXAR_CSV" \
    --nexar_data_root "$NEXAR_DATA_ROOT" \
    --dad_csv "$DAD_CSV" \
    --dad_data_root "$DAD_DATA_ROOT" \
    --dada2000_csv "$DADA2000_CSV" \
    --dada2000_data_root "$DADA2000_DATA_ROOT" \
    --dota_csv "$DOTA_CSV" \
    --dota_data_root "$DOTA_DATA_ROOT" \
    --gpu_id 0 \
    --output_dir "$output_dir" \
    --inference_batch_size 16 \
    --predict_future_temporal_steps "$PREDICT_FUTURE" \
    --classify_on_predicted_only True \
    1> >(tee "$log_file") \
    2> >(tee "$error_file" >&2)