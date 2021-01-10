#!/bin/bash

#SBATCH --job-name reward_drex
#SBATCH --output=logs/slurmjob_%j.out
#SBATCH --error=logs/slurmjob_%j.err
#SBATCH --mail-user=asaran@cs.utexas.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --partition titans
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time 72:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8

python LearnAtariSyntheticRankingsBinning.py --env_name spaceinvaders --reward_model_path ./learned_models/spaceinvaders.params --checkpoint_path ./models/spaceinvaders_25/00500 &
python LearnAtariSyntheticRankingsBinning.py --env_name breakout --reward_model_path ./learned_models/breakout.params --checkpoint_path ./models/breakout_25/00500 &
python LearnAtariSyntheticRankingsBinning.py --env_name phoenix --reward_model_path ./learned_models/phoenix.params --checkpoint_path ./models/phoenix_50/00500 &
python LearnAtariSyntheticRankingsBinning.py --env_name mspacman --reward_model_path ./learned_models/mspacman.params --checkpoint_path ./models/mspacman_50/00500 &
python LearnAtariSyntheticRankingsBinning.py --env_name centipede --reward_model_path ./learned_models/centipede.params --checkpoint_path ./models/centipede_50/00500 &
python LearnAtariSyntheticRankingsBinning.py --env_name seaquest --reward_model_path ./learned_models/seaquest.params --checkpoint_path ./models/seaquest_25/00500 &
python LearnAtariSyntheticRankingsBinning.py --env_name asterix --reward_model_path ./learned_models/asterix.params --checkpoint_path ./models/asterix_50/checkpoints/00500 &
wait
