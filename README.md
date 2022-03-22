# Updated MTAG
This repository contains code for a rewritten and updated version of [MTAG](https://github.com/jedyang97/MTAG) applied to Social-IQ using a novel Factorized Graph Neural Network Approach. Below, you will find dependency installation instructions, instructions for how to download the data and custom repositories, run scripts, and finally an overview of the code.

## Installation
### On Atlas (From Martin)
```
# install conda
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
# go through installation steps, make sure to install to /work/<andrewid>/, NOT to /home b/c of space

add these lines to your ~/.bashrc so all packages go to /work, not /home
export CONDA_ENVS_PATH='/work/<andrewid>/anaconda3/envs'
export CONDA_PKGS_DIRS='/work/<andrewid>/anaconda3/pkgs'
export PIP_CACHE_DIR='/work/<andrewid>/.cache/pip'
export XDG_CACHE_HOME='/work/<andrewid>/.cache'

# create conda environment, install torch
srun -p gpu_low --exclude compute-0-33,compute-1-37,compute-1-13,compute-0-37,compute-1-25,compute-1-9 --gres=gpu:1 --mem=56GB --pty bash
module load gcc-10.2.0
conda create -n hi python==3.7.0 -y
conda activate hi
conda install -y cudatoolkit=10.2
pip install scipy PyYAML
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
conda upgrade -c anaconda pip
p -c "import torch; print(torch.cuda.is_available())"
conda install -c conda-forge librosa
pip install -r requirements.txt
```

### On local machine
Torch-geometric is notoriously difficult to get working with different versions.  If it isn't working for you, check out their [installation page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).  Here's what works on Ubuntu 20.04.
```
export CUDA=cu102
export TORCH=1.8.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

pip install -r requirements.txt
```

## Preliminaries
First, download the data. I've uploaded it on google drive at this link. Extract it to `data/`.

```
git clone https://github.com/abwilf/Factorized.git && cd Factorized/
# Skip the following three steps if you are on the lab ATLAS machine
mkdir -p data/ && cd data/
pip install gdown
gdown https://drive.google.com/uc?id=11PNtUAjfgEre8uN83ThMewCm5qGL-qb4
```

Replace all instances of `/work/awilf/` in this codebase with your top level directory (the directory just above this one).  

```
cd ..
find . -type f -exec sed -i 's/\/work\/awilf/[YOUR DIRECTORY]/g' {} + # remember to use \/ to represent "/" in directories
```

Clone this repository into it and name it `MTAG` (`mv Factorized MTAG`), so that you have this repo as `[YOUR DIRECTORY]/MTAG` (after replace `/work/awilf/` with your top level).

```
cd ..
mv Factorized MTAG
```

You will also need to clone these two repositories into the same top level directory:
```
git clone https://github.com/abwilf/Standard-Grid
git clone https://github.com/martinmamql/CMU-MultimodalSDK.git # slightly updated version for this repo
```

## Running the Program

For social-iq, run `bash run_word.sh`. # if you are on the lab ATLAS server, use 
To run on mosi, use `bash run_mosi.py`. 

## Program Structure
`arg_defaults.py`: contains program arguments.  These are passed in `main.py` to the `gc` (global consts) variable, which is passed all around the program.  Relatively few functions rely heavily on arguments – rather, most flags are set within the `gc` object for flexibility and speed of iteration.
`main.py`: training loop for the different functions.
`models/social_iq.py`: contains low level data processing code, including alignment step.  This processed code is used directly by the baseline model, and further processed for the graph approaches.
`models/factorized.py`: this is where we construct the graphs and define the factorized model.
`models/mosi.py`: contains data processing in model creation for mosi

