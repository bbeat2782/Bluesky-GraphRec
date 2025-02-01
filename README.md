# Bluesky-GraphRec

This repository is built for the project **Bluesky: Post Recommendation**, primarily leveraging **DyGFormer** as the backbone model. We acknowledge the authors of DyGFormer for their foundational contributions, as detailed in [Towards Better Dynamic Graph Learning: New Architecture and Unified Library](https://arxiv.org/abs/2303.13047).

## Overview

TODO

## Preprocessing

The Myket dataset comes from [Effect of Choosing Loss Function when Using T-batching for Representation Learning on Dynamic Networks](https://arxiv.org/abs/2308.06862) and 
can be accessed from [here](https://github.com/erfanloghmani/myket-android-application-market-dataset). 
The original and preprocessed files for Myket dataset are included in this repository.

We can run ```preprocess_data/preprocess_data.py``` for pre-processing the datasets.
For example, to preprocess the *Bluesky* dataset, we can run the following commands:
```{bash}
cd preprocess_data/
python preprocess_data.py  --dataset_name bluesky
```

## Evaluation Tasks

DyGLib supports dynamic link prediction under both transductive and inductive settings with three (i.e., random, historical, and inductive) negative sampling strategies,
as well as dynamic node classification.


## Environments
Follow these steps to set up the `bluesky` environment:

### **1️⃣ Create the Conda Environment**
This installs all Conda-managed dependencies.
```
conda env create -f environment.yml
```
### **2️⃣ Activate the Environment**
Once the installation is complete, activate the environment.
```
conda activate bluesky
```
### **3️⃣ Install Additional Pip Packages**
Some packages, such as torch-geometric, are installed via pip. So to ensure all packages are installed correctly, run:
```
pip install -r requirements.txt
```

## Executing Scripts

### Scripts for Dynamic Link Prediction
Dynamic link prediction could be performed on all the thirteen datasets. 
If you want to load the best model configurations determined by the grid search, please set the *load_best_configs* argument to True.
#### Model Training
* Example of training *GraphRec* on *Bluesky* dataset:
```{bash}
python train_link_prediction.py --dataset_name bluesky --model_name GraphRec --patch_size 2 --max_input_sequence_length 64 --num_runs 5 --gpu 0 --batch_size 500 --negative_sample_strategy historical
```
#### Model Evaluation
Three (i.e., random, historical, and inductive) negative sampling strategies can be used for model evaluation.
* Example of evaluating *GraphRec* with posts that received at least one like in the last 20 minutes as candidate generation on *Bluesky* dataset:
```{bash}
python evaluate_link_prediction_v2.py --dataset_name bluesky --model_name GraphRec --patch_size 2 --max_input_sequence_length 64 --negative_sample_strategy real --num_runs 1 --gpu 0 --batch_size 4
```


## Acknowledgments

We are grateful to the authors of 
[TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs), 
[TGN](https://github.com/twitter-research/tgn), 
[CAWN](https://github.com/snap-stanford/CAW), 
[EdgeBank](https://github.com/fpour/DGB),
[GraphMixer](https://github.com/CongWeilin/GraphMixer), and
[DyGFormer](https://github.com/yule-BUAA/DyGLib) for making their project codes publicly available.


## Citation

```{bibtex}
@article{yu2023towards,
  title={Towards Better Dynamic Graph Learning: New Architecture and Unified Library},
  author={Yu, Le and Sun, Leilei and Du, Bowen and Lv, Weifeng},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
