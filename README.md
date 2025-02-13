# Bluesky-GraphRec

This repository is built for the project **Bluesky: Post Recommendation**, primarily leveraging **DyGFormer** as the backbone model. We acknowledge the authors of DyGFormer for their foundational contributions, as detailed in [Towards Better Dynamic Graph Learning: New Architecture and Unified Library](https://arxiv.org/abs/2303.13047).

## Overview

Bluesky-GraphRec is a dynamic post recommendation system for Bluesky, leveraging DyGFormer to model evolving user-post interactions over time. Unlike static methods, our approach incorporates graph-based recommendations with temporal learning, ensuring personalized and adaptive suggestions. We implement efficient candidate generation and evaluate performance using dynamic link prediction tasks. The system is benchmarked against models like TGAT, analyzing whether recommendations favor popular content or promote diverse engagement.

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

## Preprocessing
For extracting interactions from duckdb and converting text to text embeddings.
```{bash}
cd preprocess_data/
python extract_from_duckdb.py
```

For creating user features using SVD on a consumer-producer graph.
```{bash}
python preprocess_user_features.py
```

We can run ```preprocess_data/preprocess_data.py``` for pre-processing the datasets.
To preprocess the *Bluesky* dataset, we can run the following commands:
```{bash}
cd preprocess_data/
python preprocess_data.py  --dataset_name bluesky
```

## Evaluation Tasks

### Scripts for Dynamic Link Prediction

#### Model Training
* Training *GraphRec* on *Bluesky* dataset:
```{bash}
python train_link_prediction.py --dataset_name bluesky --model_name GraphRecMultiCo --patch_size 2 --max_input_sequence_length 64 --num_runs 1 --gpu 0 --batch_size 512 --negative_sample_strategy historical --num_epochs 50 --num_heads 2 --seed 42
```

* Training *TGAT* on *Bluesky* dataset:
```{bash}
python train_link_prediction.py --dataset_name bluesky --model_name TGAT --num_runs 1 --gpu 0 --batch_size 256 --negative_sample_strategy historical --num_epochs 50 --num_neighbors 8 --num_layers 2 --num_heads 2 --seed 42
```

#### Model Evaluation
* Evaluating *GraphRec* with posts that received at least one like in the last 20 minutes as candidate generation on *Bluesky* dataset:
```{bash}
python evaluate_link_prediction.py --dataset_name bluesky --model_name GraphRecMultiCo --patch_size 2 --max_input_sequence_length 64 --num_runs 1 --gpu 0 --batch_size 4 --negative_sample_strategy real --num_heads 2 --seed 42
```

* Evaluating *TGAT* with posts that received at least one like in the last 20 minutes as candidate generation on *Bluesky* dataset:
```{bash}
python evaluate_link_prediction.py --dataset_name bluesky --model_name TGAT --num_runs 1 --gpu 0 --batch_size 4 --negative_sample_strategy real --num_neighbors 8 --num_layers 2 --num_heads 2 --seed 42
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
