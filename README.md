# Updated MTAG
This repository contains code for a rewritten and updated version of [MTAG](https://github.com/jedyang97/MTAG) as part of ANLP 11-711's third course project.

## Installation
Torch-geometric is notoriously difficult to get working with different versions.  If it isn't working for you, check out their [installation page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
```
export CUDA=cu102
export TORCH=1.8.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

pip install -r requirements.txt
```

## Preliminaries
First, download the data. I've uploaded it on google drive at this link.  The file is 17GB. 
```
https://drive.google.com/file/d/1Iz-Ag1phSyqffWFctpVLX9yNxhYw-_8G/view?usp=sharing
```

If you'd prefer to download it from the command line, here's a script:
```
pip install gdown
gdown https://drive.google.com/uc?id=1Iz-Ag1phSyqffWFctpVLX9yNxhYw-_8G
tar -xvf deployed.tar
```

Replace all instances of `/work/awilf/` in this codebase with your top level directory (the directory just above this one).  I'm sure there's some way to do this via the command line; I use vscode for it. Clone this repository into it and name it `MTAG` (`mv mtag MTAG`), so that you have this repo as `/work/awilf/MTAG` (replace `/work/awilf/` with your top level).

You will also need to clone these two repositories into the same top level directory:
```
https://github.com/abwilf/Standard-Grid
https://github.com/A2Zadeh/CMU-MultimodalSDK
```

Your top level directory (e.g. `/work/awilf`) should look something like this:
```
$ tree -L 1
├── CMU-MultimodalSDK
├── MTAG
├── Standard-Grid
...
```

Make sure the `deployed` folder is within `MTAG`, and now you should be able to run the program.
```
git checkout het
bash run.sh
```
