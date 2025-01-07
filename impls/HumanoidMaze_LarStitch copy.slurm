#!/bin/bash
#SBATCH --job-name=HM_LarSR
#SBATCH --nodes=5                # node count
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=8G   # memory per cpu-core
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=36:00:00
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ij9461@princeton.edu
#SBATCH --output=slurm_logs/%j.log    # Single log file for both stdout and stderr
#SBATCH --error=slurm_logs/%j.log     # Same file as output


module purge
module load anaconda3/2024.2
conda activate ogbench

wandb online

# Define variables for the experiment
ENV_NAME="humanoidmaze-large-stitch-v0"
WANDB_PROJECT_NAME="Humanoid-Maze-Large-Stitch"
RUN_GROUP="JAXGCRL"
WANDB_MODE="offline"
EVAL_EPISODES=50
AGENT_PATH="agents/jaxgcrl.py"

AGENT_ALPHA=0.1
AGENT_DISCOUNT=0.995

# Network architecture parameters for JAX GCRL
AGENT_NETWORK_DEPTH=3
AGENT_SKIP_CONNECTION_FREQUENCY=3
AGENT_NETWORK_WIDTH=512
AGENT_USE_RELU=0
AGENT_RESNET_TYPE="resnetOrig"

AGENT_LATENT_DIM=64
AGENT_STATE_DEP_STD="False"
AGENT_CONST_STD="True"

SEED=-1
NUM_SEEDS=2

# Print experiment details
echo "Environment: ${ENV_NAME}"
echo "Evaluation Episodes: ${EVAL_EPISODES}"
echo "Agent Path: ${AGENT_PATH}"
echo "Agent Alpha: ${AGENT_ALPHA}"
echo "Agent Discount: ${AGENT_DISCOUNT}"
echo "Network Depth: ${AGENT_NETWORK_DEPTH}"
echo "Skip Connection Frequency: ${AGENT_SKIP_CONNECTION_FREQUENCY}"
echo "Number of seeds: ${NUM_SEEDS}"

# Run experiments
for NUM in $(seq 1 ${NUM_SEEDS}); do
    echo "Running seed ${SEED}"
    python main.py \
        --env_name=${ENV_NAME} \
        --eval_episodes=${EVAL_EPISODES} \
        --agent=${AGENT_PATH} \
        --agent.alpha=${AGENT_ALPHA} \
        --agent.discount=${AGENT_DISCOUNT} \
        --agent.network_depth=${AGENT_NETWORK_DEPTH} \
        --agent.skip_connection_frequency=${AGENT_SKIP_CONNECTION_FREQUENCY} \
        --agent.network_width=${AGENT_NETWORK_WIDTH} \
        --agent.use_relu=${AGENT_USE_RELU} \
        --agent.resnet_type=${AGENT_RESNET_TYPE} \
        --agent.latent_dim=${AGENT_LATENT_DIM} \
        --agent.state_dependent_std=${AGENT_STATE_DEP_STD} \
        --agent.const_std=${AGENT_CONST_STD} \
        --wandb_project_name=${WANDB_PROJECT_NAME} \
        --run_group=${RUN_GROUP} \
        --wandb_mode=${WANDB_MODE} \
        --seed=${SEED}
done