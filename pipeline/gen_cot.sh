num_slots=1
HOSTFILE="${MPI_HOSTFILE:?Set MPI_HOSTFILE to your MPI hostfile}"
sed -i "s/slots=[0-9]\+\$/slots=$num_slots/g" "$HOSTFILE"

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

export LD_PRELOAD="${NCCL_LIB_PATH:?Set NCCL_LIB_PATH to libnccl.so.2}"
mpirun -x LD_PRELOAD -x NCCL_DEBUG=INFO --hostfile "$HOSTFILE" python gen_cot.py
