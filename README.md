# CALVIN RLDS Dataset Conversion
Reference repo: https://github.com/kpertsch/rlds_dataset_builder, thanks a lot.


This repo contatins the scripts to convert [CALVIN](https://github.com/mees/calvin) dataset into [RLDS](https://github.com/google-research/rlds) format for X-embodiment experiment integration.

## How to use the script

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

If you want to manually create an environment, the key packages to install are `tensorflow`, 
`tensorflow_datasets`, `tensorflow_hub`, `apache_beam`, `matplotlib`, `plotly` and `wandb`.






**Thanks a lot for contributing your data! :)**
