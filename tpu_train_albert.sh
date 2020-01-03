export STORAGE_BUCKET=gs://nqa-data
export DATA_DIR="$STORAGE_BUCKET/data"
export MODELS_DIR="models/"
export ALBERT_SIZE="xxl"

python3 tpu_train.py \
--model="albert" \
--config_file="$MODELS_DIR/albert_$ALBERT_SIZE/config.json" \
--vocab_file="$MODELS_DIR/albert_$ALBERT_SIZE/vocab/modified-30k-clean.model" \
--output_dir="output/" \
--train_precomputed_file="$DATA_DIR/albert_train_small.tf_record" \
--train_num_precomputed=-1 \
--output_checkpoint_file="albert_finetuned.h5" \
--save_checkpoints_steps=15000 \
--log_dir=gs://tmp-log-data \
--log_freq=128 \
--do_train=True \
--do_predict=False \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=3 \
--train_file="$DATA_DIR/simplified-nq-train.jsonl" \
--albert_pretrain_checkpoint="$MODELS_DIR/albert_$ALBERT_SIZE/tf2_model.h5" \
--use_tpu=True \
--tpu_name=$TPU_NAME \
--tpu_zone="us-central1-f" \
--gcp_project="nqa-tpu-training"
