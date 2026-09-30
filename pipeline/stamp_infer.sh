export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

mpirun -np 8 \
    -x VLLM_ATTENTION_BACKEND=FLASHINFER \
    python AV_segment_caption.py \
    --input-jsonl path/to/data.jsonl \
    --output-dir path/to/output \
    --audio-dir path/to/audio