#!/bin/bash
#
#SBATCH -p gpu_low
#SBATCH --gres=gpu:1  # Use GPU
#SBATCH --mem 56GB   # memory pool for all cores
#SBATCH -t 1-00:00    # time (D-HH:MM)
#SBATCH -o /work/awilf/MTAG/results/%j.out        # STDOUT. %j specifies JOB_ID.
#SBATCH -e /work/awilf/MTAG/results/%j.err        # STDERR. See the first link for more options.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dummyblah123@gmail.com
#SBATCH --exclude=compute-0-33,compute-1-37,compute-1-13,compute-0-37,compute-1-25,compute-1-9

# run this with: sbatch -p gpu_low ./blah.sh
# echo "hi"
# sleep 2
# source activate fairseq

cd /work/awilf/MTAG

ulimit -v unlimited
singularity exec --nv -B /work/awilf/awilf/.local/python3.7/site-packages:/home/awilf/.local/lib/python3.7/site-packages,/work/awilf/MTAG,/work/awilf/Standard-Grid,/work/awilf/CMU-MultimodalSDK,/work/awilf/Social-IQ blah4.sif bash run.sh


