export DATA_DIR="data"
export MODELS_DIR="models"
export ALBERT_SIZE="xl"

python3 tpu_train.py \
--model="albert" \
--config_file="$MODELS_DIR/albert_$ALBERT_SIZE/config.json" \
--vocab_file="$MODELS_DIR/albert_$ALBERT_SIZE/vocab/modified-30k-clean.model" \
--output_dir="output/" \
--train_precomputed_file="$DATA_DIR/albert_train.tf_record" \
--train_num_precomputed=-1 \
--output_checkpoint_file="albert_finetuned.h5" \
--save_checkpoints_steps=5000 \
--log_dir="logs/" \
--log_freq=1 \
--do_train=True \
--do_predict=False \
--train_batch_size=1 \
--learning_rate=5e-5 \
--num_train_epochs=3 \
--train_file="$DATA_DIR/simplified-nq-train.jsonl" \
--init_checkpoint="$MODELS_DIR/albert_$ALBERT_SIZE/tf2_model_pretrain.h5" \
--use_tpu=False