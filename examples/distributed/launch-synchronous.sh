export EPOCHS=1
export LEARNING_RATE=0.01
export MOMENTUM=.9
export NUM_NODES=4
export NUM_PROCS_NODE=1
export SCRIPT="synchronous.py"
export NODES=$(cat nodes | head -n $NUM_NODES)
export MASTER_ADDRESS=$(echo $NODES | shuf | awk '{print $1}')
export MASTER_ADDRESS=$(host $MASTER_ADDRESS | head -n 1 | awk '{print $4}')
export MASTER_PORT=1234
export BATCH_SIZE=5
export DISTRIBUTED_FS=1

# Retrieve the user's password and username for SSH authentication.
read -sp "Username: " USERNAME
echo ""
read -sp "Password: " PASSWORD
echo ""

# Copy the files to the required machines.
# Check if a distributed filesystem is present.
if [ $DISTRIBUTED_FS -eq 1 ]; then
    # Copy the files once to the primary node.
    NODE=$(cat nodes | head -n 1)
    # Copy the required files.
    sshpass -p $PASSWORD scp -oStrictHostKeyChecking=no "model.py" $USERNAME@$NODE:~/"model.py"
    sshpass -p $PASSWORD scp -oStrictHostKeyChecking=no $SCRIPT $USERNAME@$NODE:~/$SCRIPT
else
    # Copy the required files to all nodes.
    for NODE in $NODES; do
        # Copy the required files.
        sshpass -p $PASSWORD scp -oStrictHostKeyChecking=no "model.py" $USERNAME@$NODE:~/"model.py"
        sshpass -p $PASSWORD scp -oStrictHostKeyChecking=no $SCRIPT $USERNAME@$NODE:~/$SCRIPT
    done
fi

NODE_RANK=0
for NODE in $NODES; do
    # Initiate the optimization procedure at the current node.
    sshpass -p $PASSWORD ssh -oStrictHostKeyChecking=no $USERNAME@$NODE "\
        killall -9 python; \
        source .bashrc; \
        export BATCH_SIZE=$BATCH_SIZE; \
        python -m torch.distributed.launch \
            --nproc_per_node=$NUM_PROCS_NODE \
            --nnodes=$NUM_NODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDRESS \
            --master_port=$MASTER_PORT $SCRIPT \
    " &
    # Increase the rank for the next node.
    NODE_RANK=$(($NODE_RANK + 1))
done

# Wait for all processes to complete.
wait
