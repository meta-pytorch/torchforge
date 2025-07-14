set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./configs/llama3_8b.yaml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m apps.sft.sft --job.config_file ${CONFIG_FILE} $overrides