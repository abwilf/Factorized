# Updated MTAG

This repository contains code for a rewritten and updated version of [MTAG](https://github.com/jedyang97/MTAG) as part of ANLP 11-711's third course project.  

## Reproducing
To run our code, simply clone this repository (please make sure you're on the branch `het`), then follow the installation instructions from the [original codebase](https://github.com/jedyang97/MTAG) for the various packages, depending on your GPU setting.  Then run `bash run.sh` for a simple run of the code.  To reproduce any of our hyperparameter search results, you can susbtitute any of the values in `{iemocap/mosi/social}_grid_search.csv` into `run.sh`.

To run the original code for MTAG on MOSI and IEMOCAP, please clone their [repository](https://github.com/jedyang97/MTAG) and use the scripts we attached in our project upload, `run_their_iemocap.sh` and `run_their_mosi.sh` to run their code with their best hyperparameters.

