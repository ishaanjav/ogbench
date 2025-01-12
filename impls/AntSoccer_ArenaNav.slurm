#!/bin/bash
#SBATCH --job-name=AC_416
#SBATCH --nodes=5                # node count
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=16G   # memory per cpu-core
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=11:59:59
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
RUN_GROUP="Cold"
WANDB_MODE="offline"
EVAL_EPISODES=50
AGENT_PATH="agents/crl.py"

ACTOR_NUM_HIDDEN_LAYERS=4
NUM_HIDDEN_LAYERS=16




SKIP_CONNECTION_FREQUENCY=4
ACTOR_SKIP_CONNECTION_FREQUENCY=4


ACTIVATION_FN="gelu"  # Options: "gelu", "swish"
KERNEL_INIT="default"  # Options: "default", "lecun_uniform"

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
        --agent.activation_fn=${ACTIVATION_FN} \
        --agent.kernel_init=${KERNEL_INIT} \
        --agent.num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        --agent.actor_num_hidden_layers=${ACTOR_NUM_HIDDEN_LAYERS} \
        --agent.skip_connection_frequency=${SKIP_CONNECTION_FREQUENCY} \
        --agent.actor_skip_connection_frequency=${ACTOR_SKIP_CONNECTION_FREQUENCY} \
        --agent.cold_initialization=True \
        --seed=${SEED}
done