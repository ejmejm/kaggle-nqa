export STORAGE_BUCKET=gs://nqa-data-new
export DATA_DIR="$STORAGE_BUCKET/data"
export MODELS_DIR="models"
export ALBERT_SIZE="xxl"

python3 tpu_train_verifier.py \
--model="albert" \
--config_file="$MODELS_DIR/albert_$ALBERT_SIZE/config.json" \
--vocab_file="$MODELS_DIR/albert_$ALBERT_SIZE/vocab/modified-30k-clean.model" \
--output_dir="output/" \
--train_precomputed_file="$DATA_DIR/verifier_train.tf_record" \
--train_num_precomputed=-1 \
--output_checkpoint_file="verifier_model.h5" \
--save_checkpoints_steps=20000 \
--log_dir=gs://tmp-log-data/albert-logs/verifier/ \
--log_freq=32 \
--do_train=True \
--do_predict=False \
--train_batch_size=16 \
--learning_rate=5e-5 \
--num_train_epochs=3 \
--train_file="$DATA_DIR/simplified-nq-train.jsonl" \
--init_checkpoint="$MODELS_DIR/albert_$ALBERT_SIZE/tf2_model_pretrain.h5" \
--use_tpu=True \
--tpu_name=tpu-verifier \
--tpu_zone="us-central1-f" \
--gcp_project="nqa-tpu-training"
