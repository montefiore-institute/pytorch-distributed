export EPOCHS=200
export LEARNING_RATE=0.01
export MOMENTUM=.9
export NUM_NODES=11
export NUM_PROCS_NODE=1
export SCRIPTS="asynchronous.py util.py model.py"
export MAIN_SCRIPT="asynchronous.py"
export NODES=$(cat nodes | shuf | head -n $NUM_NODES)
export MASTER_ADDRESS=$(echo $NODES | awk '{print $1}')
export MASTER_PORT=1234
export BATCH_SIZE=64
export DISTRIBUTED_FS=1

# Retrieve the user's password and username for SSH authentication.
read -sp "Username: " USERNAME
echo ""
read -sp "Password: " PASSWORD
echo ""

function copy_scripts {
    for SCRIPT in $SCRIPTS; do
        sshpass -p $PASSWORD scp -oStrictHostKeyChecking=no $SCRIPT $USERNAME@$1:~/$SCRIPT
    done
}

# Copy the files to the required machines.
if [ $DISTRIBUTED_FS -eq 1 ]; then
    # Copy the files once to the primary node.
    NODE=$(cat nodes | head -n 1)
    copy_scripts $NODE
else
    # Copy the files to all machines.
    for NODE in $NODES; do
        copy_scripts $NODE
    done
fi


NODE_RANK=0
for NODE in $NODES; do
    sshpass -p $PASSWORD ssh -oStrictHostKeyChecking=no $USERNAME@$NODE "\
        killall -9 python; \
        source .bashrc; \
        python -m torch.distributed.launch \
            --nproc_per_node=$NUM_PROCS_NODE \
            --nnodes=$NUM_NODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDRESS \
            --master_port=$MASTER_PORT $MAIN_SCRIPT \
            --batch-size $BATCH_SIZE \
            --epochs $EPOCHS \
            --lr $LEARNING_RATE \
            --momentum $MOMENTUM \
            --download \
            --downloadm
    " &
    NODE_RANK=$(($NODE_RANK + 1))
done

wait
