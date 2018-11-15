## Introduction
This repository is an extension of the [EEL](https://github.com/henryzord/eel), in which performing a stacking with a logistic regression as a meta-classifier  and as base-classifier EEL.

## Testing
The experiments were conducted on a 10-fold cross-validation. We use a seed (defined in file ```params.json```) for partitioning the datasets into folds. By using ```random_state=0```, you will guarantee that the folds used by your algorithm are the same as the ones used by EEL.

We do not, however, set a seed for our stochastic algorithm to run, so expect slightly different results from EEL as the ones reported in the paper.

#### Setup

We provide a tutorial on how to run experiments based on the [Anaconda](https://www.anaconda.com/download/#linux) distribution of Python, with the Linux OS. Once installed, create a virtual environment for the experiments:

```
conda create --name env_eel python=3.6 --yes
```
Activate the environment using 
```
source activate env_eel
```
Install requirements from the file with
```
pip install -r requirements.txt
```
Finally, create a folder for meta data using 
```
mkdir metadata
```
You may have to create a specific folder for each tested algorithm.

For testing EEL, simply run a command like in the following example:
```
python test_eel.py -d "/home/user/datasets" -m "/home/user/metadata" -p "/home/user/params.json" --n_run 10
``` 
with required parameters. 

Finally, The folder ```visual``` has several graphical ammenities used for generating figures in the paper. 