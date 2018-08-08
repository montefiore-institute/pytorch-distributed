export EPOCHS=1
export LEARNING_RATE=0.01
export MOMENTUM=.9
export NUM_NODES=5
export NUM_PROCS_NODE=1
export SCRIPT="asynchronous.py"
export NODES=$(cat nodes | head -n $NUM_NODES)
export USERNAME="joherman"
export MASTER_ADDRESS=$(echo $NODES | awk '{print $1}')
export MASTER_ADDRESS=$(host $MASTER_ADDRESS | head -n 1 | awk '{print $4}')
export MASTER_PORT=1234
export BATCH_SIZE=50

# Retrieve the user's password and username for SSH authentication.
read -sp "Username: " USERNAME
echo ""
read -sp "Password: " PASSWORD
echo ""

NODE_RANK=0
for NODE in $NODES; do
    # Kill all the Python processes on the machine.
    sshpass -p $PASSWORD ssh $USERNAME@$NODE "killall -9 python" > /dev/null
    # Copy the script and model file.
    sshpass -p $PASSWORD scp "model.py" $USERNAME@$NODE:~/"model.py"
    sshpass -p $PASSWORD scp $SCRIPT $USERNAME@$NODE:~/$SCRIPT
    # Initiate the optimization procedure at the current node.
    sshpass -p $PASSWORD ssh $USERNAME@$NODE "\
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
