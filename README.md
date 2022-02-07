# Updated MTAG
This repository contains code for a rewritten and updated version of [MTAG](https://github.com/jedyang97/MTAG) applied to Social-IQ using a novel Factorized Graph Neural Network Approach. Below, you will find dependency installation instructions, instructions for how to download the data and custom repositories, run scripts, and finally an overview of the code.

## Installation
Torch-geometric is notoriously difficult to get working with different versions.  If it isn't working for you, check out their [installation page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).  Here's what works on Atlas and Ubuntu 20.04.
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
https://drive.google.com/file/d/12OKD-h4WfrjXMeBlRJbHDiYUEtAWfG-1/view?usp=sharing
```

Replace all instances of `/work/awilf/` in this codebase with your top level directory (the directory just above this one).  I'm sure there's some way to do this via the command line; I use vscode for it. Clone this repository into it and name it `MTAG` (`mv mtag MTAG`), so that you have this repo as `/work/awilf/MTAG` (replace `/work/awilf/` with your top level).

You will also need to clone these two repositories into the same top level directory:
```
https://github.com/abwilf/Standard-Grid
https://github.com/A2Zadeh/CMU-MultimodalSDK
```

Make sure the `deployed` folder is within `MTAG`, and now you should be able to run the program.
```
git checkout het
bash run.sh
```

## Running the Program
To run the `chunk` version of the program up to SOTA performance, run `bash run_chunk.sh`.  Likewise with the `word` version, run `bash run_word.sh`. 

To run on mosi, use `bash run_mosi.py`. 

## Program Structure
`arg_defaults.py`: contains program arguments.  These are passed in `main.py` to the `gc` (global consts) variable, which is passed all around the program.  Relatively few functions rely heavily on arguments – rather, most flags are set within the `gc` object for flexibility and speed of iteration.
`main.py`: training loop for the different functions.
`models/social_iq.py`: contains low level data processing code, including alignment step.  This processed code is used directly by the baseline model, and further processed for the graph approaches.
`models/factorized.py`: this is where we construct the graphs and define the factorized model.
`models/mosi.py`: contains data processing in model creation for mosi

