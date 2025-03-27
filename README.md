# CALVIN RLDS Dataset Conversion
Reference repo: https://github.com/kpertsch/rlds_dataset_builder, thanks a lot.


This repo contatins the scripts to convert [CALVIN](https://github.com/mees/calvin) dataset into [RLDS](https://github.com/google-research/rlds) format for X-embodiment experiment integration.

## How to use the script

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Second activate the environment using:
```
conda activate rlds_env
```
Change the dataset dir to your local disk in this [line](https://github.com/hyy02/calvin_rlds_builder/blob/a83ea80b890d2e9a4f1fee02ca1a7c99c5254192/calvin/calvin_dataset_builder.py#L96).

Then run tfds build in the calvin dir:
```
cd calvin
tfds build --data-dir <your_path>
```
Finally, you can find the rlds type datasets in <your_path>.

If you meet some problems, feel free to raise an issue,or contact huangyiyang24@mails.ucas.ac.cn





**Thanks a lot for contributing your data! :)**
