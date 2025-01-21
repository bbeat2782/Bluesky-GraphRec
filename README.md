# Bluesky-GraphRec

This repository is built for the project **Bluesky: Post Recommendation**, primarily leveraging **DyGFormer** as the backbone model. We acknowledge the authors of DyGFormer for their foundational contributions, as detailed in [Towards Better Dynamic Graph Learning: New Architecture and Unified Library](https://arxiv.org/abs/2303.13047).

## Overview

TODO

## Preprocessing

The Myket dataset comes from [Effect of Choosing Loss Function when Using T-batching for Representation Learning on Dynamic Networks](https://arxiv.org/abs/2308.06862) and 
can be accessed from [here](https://github.com/erfanloghmani/myket-android-application-market-dataset). 
The original and preprocessed files for Myket dataset are included in this repository.

We can run ```preprocess_data/preprocess_data.py``` for pre-processing the datasets.
For example, to preprocess the *Wikipedia* dataset, we can run the following commands:
```{bash}
cd preprocess_data/
python preprocess_data.py  --dataset_name bluesky
```
We can also run the following commands to preprocess all the original datasets at once:
```{bash}
cd preprocess_data/
python preprocess_all_data.py
```

## Evaluation Tasks

DyGLib supports dynamic link prediction under both transductive and inductive settings with three (i.e., random, historical, and inductive) negative sampling strategies,
as well as dynamic node classification.


## Incorporate New Datasets or New Models

New datasets and new models are welcomed to be incorporated into DyGLib by pull requests.
* For new datasets: The format of new datasets should satisfy the requirements in ```DG_data/DATASETS_README.md```. 
  Users can put the new datasets in ```DG_data``` folder, and then run ```preprocess_data/preprocess_data.py``` to get the processed datasets.
* For new models: Users can put the model implementation in  ```models``` folder, 
  and then create the model in ```train_xxx.py``` or ```evaluate_xxx.py``` to run the model.


## Environments

TODO


## Executing Scripts

### Scripts for Dynamic Link Prediction
Dynamic link prediction could be performed on all the thirteen datasets. 
If you want to load the best model configurations determined by the grid search, please set the *load_best_configs* argument to True.
#### Model Training
* Example of training *GraphRec* on *Bluesky* dataset:
```{bash}
python train_link_prediction.py --dataset_name bluesky --model_name GraphRec --patch_size 2 --max_input_sequence_length 64 --num_runs 5 --gpu 0
```
* If you want to use the best model configurations to train *DyGFormer* on *Wikipedia* dataset, run
```{bash}
python train_link_prediction.py --dataset_name bluesky --model_name GraphRec --load_best_configs --num_runs 5 --gpu 0
```
#### Model Evaluation
Three (i.e., random, historical, and inductive) negative sampling strategies can be used for model evaluation.
* Example of evaluating *DyGFormer* with *random* negative sampling strategy on *Wikipedia* dataset:
```{bash}
python evaluate_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --negative_sample_strategy random --num_runs 5 --gpu 0
```
* If you want to use the best model configurations to evaluate *DyGFormer* with *random* negative sampling strategy on *Wikipedia* dataset, run
```{bash}
python evaluate_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --negative_sample_strategy random --load_best_configs --num_runs 5 --gpu 0
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
