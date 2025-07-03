# Probabilistic data-driven melt pool models
This is the official implementation of the paper "[Probabilistic Data-Driven Modeling of a Melt Pool
in Laser Powder Bed Fusion Additive Manufacturing](https://ieeexplore.ieee.org/document/10632066)" published on IEEE Transactions on Automation Science and Engineering (TASE).
 
Please feel free to ask any questions about this work! 

# Introduction
The widespread adoption of laser powder bed fusion (LPBF) additive manufacturing is hampered by process unreliability problems. Modeling the melt pool behavior in LPBF is crucial to develop process control methods. While data-driven models linking melt pool dynamics to specific process parameters have shown appreciable advancements, existing models often oversimplify these relationships as deterministic, failing to account for the inherent instability of LPBF processes. Such simplifications can lead to overconfident and unreliable predictions, potentially resulting in erroneous process decisions. To address this critical  issue, we propose a probabilistic data-driven approach to melt pool modeling that incorporates process noise and uncertainty. Our framework formulates a problem that includes distribution approximation and uncertainty quantification. Specifically, the Gaussian distribution with higher order priors, aided with variational inference and importance sampling, is used to approximate the probability distribution of melt pool characteristics. The uncertainty inherent in both LPBF process data and the modeling approach itself are then decomposed and approximated by using Monte Carlo sampling. The melt pool model is improved further by using a novel grid-based representation for the neighborhood of a fusion point, and a neural network architecture designed for effective feature fusion. This approach not only refines the accuracy of the model but also quantifies the uncertainty of the predictions, thereby enabling more informed decision-making with reduced risk. Two potential applications, including LPBF process planning and anomaly detection, are discussed.

# Installation
It is suggested to use Conda to configure Python environments.
First git clone the project:
```
git clone https://github.com/qihangGH/probabilistic_melt_pool_model.git
```
Then change the directory to `probabilistic_melt_pool_model`:
```
cd probabilistic_melt_pool_model
```
Create the virtual environment:
```
conda env create -f prob_mp.yaml
```
Activate the environment:
```
conda activate prob_mp
```

# Data
Download the melt pool monitoring dataset released by the National Institute of Standards and Technology (NIST) 
at https://data.nist.gov/od/id/85196AB9232E7202E053245706813DFA2044, including `In-situ Meas Data.zip` and `Build Command Data.zip`. 
A dataset description article is provided at https://nvlpubs.nist.gov/nistpubs/jres/124/jres.124.033.pdf.  

In `Build Command Data.zip`, unzip `XYPT Commands` folder, and 
in `In-situ Meas Data.zip`, unzip the `Melt Pool Camera` folder.

# Preprocessing
## Extract melt pool size
```
python mp_feature_extraction.py \
    --data_dir <dir_to_Melt_Pool_Camera> \
    --save_filepath <filepath_to_save_results>
```

## Prepare the grid-based neighborhood dataset
First run
```
python datasets/grid_based_data.py \
    --original_data <dir_to_Build_command_data> \
    --melt_pool_feature_path <filepath_to_mp_feature> \ 
    --save_path <path_to_save_results>
```
You can change the layers to be processed in `datasets/grid_based_data.py`.
Then generate the dataset by
```
python datasets/generate_grid_based_datasets.py \
    --root_save_dir <root_dir_to_save_data> \
    --patch_data_path <patch_data_generated_by_grid_based_data.py>
```
where `root_save_dir` is the folder to save the dataset, and `patch_data_path` is the same as the `save_path`
set in `grid_based_data.py`.

# Training
```
python train_mp_models.py \
    --data_dir <directory_of_the_dataset> \
    --save_dir <directory_to_save_training_results> \
    --model_name <model_name>
```
where `data_dir` is the directory of the dataset, which is the same as `root_save_dir`
set in `generate_grid_based_datasets.py`, `save_dir` is the directory to save training
results, and `model_name` can be `non_bayesian`, `gaussian`, or `student`. The `non_bayesian`
indicates a deterministic model, `gaussian` indicates a probabilistic model with the Gaussian distribution,
and `student` is the Gaussian distribution with higher order prior, resulting in a Student's t-distribution.
There are many other settings about the model structure, learning rate, etc. You can refer to 
`training_mp_models.py` for more details.

# Testing
```
python test_mp_models.py \
    --save_dir <directory_where_training_results_saved>
```
where `save_dir` is the same as the one set in `train_mp_models.py`.

# Citation
If you find our work and this repository helpful, please consider citing our papers:
```
@article{fang2024probabilistic,
  author={Fang, Qihang and Xiong, Gang and Zhao, Meihua and Tamir, Tariku Sinshaw and Shen, Zhen and Yan, Chao-Bo and Wang, Fei-Yue},
  journal={IEEE Trans. Autom. Sci. Eng.}, 
  title={Probabilistic Data-Driven Modeling of a Melt Pool in Laser Powder Bed Fusion Additive Manufacturing}, 
  year={2024},
  pages={1--18},
  doi={10.1109/TASE.2024.3412431}
}
```
Based on melt pool models developed in this repository, we further develop an [uncertainty-aware parameter optimization method](https://github.com/qihangGH/uncertainty_aware_param_optim_for_AM):
```
@article{fang2025uncertainty,
  author={Fang, Qihang and Xiong, Gang and Wang, Fang and Shen, Zhen and Dong, Xisong and Wang, Fei-Yue},
  journal={IEEE Trans. Autom. Sci. Eng.}, 
  title={Uncertainty-Aware Parameter Optimization for Reliable Laser Powder Bed Fusion Additive Manufacturing},
  year={2025},
  pages={1--16}
}
```
You may also be interested in our review paper about [process Monitoring, diagnosis and control of additive manufacturing](https://ieeexplore.ieee.org/document/9950053):
```
@article{fang2022process,
  author={Fang, Qihang and Xiong, Gang and Zhou, MengChu and Tamir, Tariku Sinshaw and Yan, Chao-Bo and Wu, Huaiyu and Shen, Zhen and Wang, Fei-Yue},
  journal={IEEE Trans. Autom. Sci. Eng.}, 
  title={Process Monitoring, Diagnosis and Control of Additive Manufacturing}, 
  year={2024},
  volume={21},
  number={1},
  pages={1041--1067},
  doi={10.1109/TASE.2022.3215258}
}
```
