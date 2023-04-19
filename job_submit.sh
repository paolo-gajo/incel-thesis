#!/bin/sh
#SBATCH -J BERT_label
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=infinite
#SBATCH --output=/home/pgajo/working/output/MLM-mBERT.o%j
#SBATCH --error=/home/pgajo/working/output/MLM-mBERT.e%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paolo.gajo@gmail.com

cd /home/pgajo/working/src
pipenv run python /home/pgajo/working/src/2023-04-18_train_BERT_regression_hftrainer.py