export STORAGE_BUCKET=gs://nqa-data
export DATA_DIR="$STORAGE_BUCKET/data"
export MODELS_DIR="models"

python3 tpu_train.py \
--model="bert" \
--config_file="$MODELS_DIR/bert_joint_baseline/bert_config.json" \
--vocab_file="$MODELS_DIR/bert_joint_baseline/vocab-nq.txt" \
--output_dir="output/" \
--train_precomputed_file="$DATA_DIR/bert_train.tf_record" \
--train_num_precomputed=-1 \
--output_checkpoint_file="bert_finetuned.h5" \
--save_checkpoints_steps=20000 \
--log_dir=gs://tmp-log-data/bert-logs \
--log_freq=32 \
--do_train=True \
--do_predict=False \
--train_batch_size=16 \
--learning_rate=5e-7 \
--num_train_epochs=2 \
--train_file="$DATA_DIR/simplified-nq-train.jsonl" \
--init_checkpoint="gs://nqa-data/bert_joint_baseline/tf2_bert_joint.ckpt" \
--use_tpu=True \
--tpu_name=$TPU_NAME \
--tpu_zone="us-central1-f" \
--gcp_project="nqa-tpu-training"
