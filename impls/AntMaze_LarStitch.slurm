#!/bin/bash
#SBATCH --job-name=AM_LarS
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
ENV_NAME="antmaze-large-stitch-v0"
WANDB_PROJECT_NAME="Ant-Maze-Large-Stitch"
RUN_GROUP="CRL"
WANDB_MODE="offline"
EVAL_EPISODES=50
AGENT_PATH="agents/crl.py"

AGENT_ALPHA=0.1
AGENT_DISCOUNT=0.99

SEED=-1
NUM_SEEDS=2

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
        --seed=${SEED} \
        --agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5 \
done