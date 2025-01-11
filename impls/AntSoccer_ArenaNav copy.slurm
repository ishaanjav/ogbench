#!/bin/bash
#SBATCH --job-name=AS_C8
#SBATCH --nodes=5                # node count
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=16G   # memory per cpu-core
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=15:59:59
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
ENV_NAME="antsoccer-arena-navigate-v0"
WANDB_PROJECT_NAME="Ant-Soccer-Arena-Navigate"
RUN_GROUP="RESNET-4"
WANDB_MODE="offline"
EVAL_EPISODES=50
AGENT_PATH="agents/crl.py"

NUM_HIDDEN_LAYERS=8  # Add this line

AGENT_ALPHA=0.3
AGENT_DISCOUNT=0.99

SEED=-1
NUM_SEEDS=1

# Print experiment details
echo "Environment: ${ENV_NAME}"
echo "Evaluation Episodes: ${EVAL_EPISODES}"
echo "Agent Path: ${AGENT_PATH}"
echo "Agent Alpha: ${AGENT_ALPHA}"
echo "Agent Discount: ${AGENT_DISCOUNT}"
echo "Number of seeds: ${NUM_SEEDS}"


# Run first experiment with variables
for NUM in $(seq 1 ${NUM_SEEDS}); do
    echo "Running seed ${SEED}"
    python main.py \
        --env_name=${ENV_NAME} \
        --eval_episodes=${EVAL_EPISODES} \
        --agent=${AGENT_PATH} \
        --agent.alpha=${AGENT_ALPHA} \
        --agent.discount=${AGENT_DISCOUNT} \
        --wandb_project_name=${WANDB_PROJECT_NAME} \
        --run_group=${RUN_GROUP} \
        --wandb_mode=${WANDB_MODE} \
        --agent.use_resnet=True \
        --agent.num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        --seed=${SEED}
done