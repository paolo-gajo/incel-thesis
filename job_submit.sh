#!/bin/sh
#SBATCH -J MLM-mBERT
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=infinite
#SBATCH --output=MLM-mBERT.o%j
#SBATCH --error=MLM-mBERT.e%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paolo.gajo@gmail.com

cd /home/pgajo/working/src
pipenv run python /home/pgajo/working/src/2023-03-27-mbert-mlm.py
