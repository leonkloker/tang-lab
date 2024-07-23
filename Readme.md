# Repository Overview

This repository contains the following files:

- `./data`: Contains the raw csv data as well as all datasets that are created using functions from data.py such
as regular train/test splits, kfold cross-validation and datasets with marginal distribution features
- `./figures`: Contains result plots from the marginal models (features sampled from estimated marginal distributions), the moment model (features are statistical characteristics of the cell values) and ablation plots for the final paper
- `./results`: Contains the model performance summaries for different training runs
- `autoencoder.py`: Contains the implementation and training loop for a simple single-cell feedforward autoencoder
compressing the 17 cell features to a low-dimensional latent space.
- `cluster_individual_cells.py`: Contains a script to cluster cells using KMeans on their raw feature values or
their autoencoder latent-space representation and to compare the ratio of clusters to the activation rates.
- `data.py`: Contains a variety of functions related to loading raw population data from a csv file, creating 
populations, augmenting them, calculating different features on each population, creating a cross-validation
dataset as well as building dataloaders for the autoencoder or neural network.
- `evaluation.py`: Contains functions that can be used to calculate evaluation metrics for regression
and classification as well as to visualize the model performance.
- `finetune_ridge_reg_param.py`: Contains a script to finetune the ridge regression parameter
(obsolete as final model is ordinary linear regression)
- `finetune_sampling_points.py`: Contains a script to finetune the amount of sampling
points when sampling from the feature's marginal distributions
(obsolete as final model does not work with marginal distributions)
- `finetune_sampling_points.py`: Contains a script to finetune the sampling range
when sampling from the feature's marginal distributions
(obsolete as final model does not work with marginal distributions)
- `model_marginal_kfold.py`: Contains a script to train and evaluate the marginal distribution model
on a kfold cross-validation dataset (obsolete as final model does not work with marginal distributions)
- `model_marginal.py`: Contains a script to train and evaluate the marginal distribution model
on a normal train/test split dataset (obsolete as final model does not work with marginal distributions)
- `model_momemnts_kfold.py`: Contains a script to train and evaluate the model using statistical moments
or characteristics as features on a kfold cross-validation dataset
- `model_moments.py`: Contains a script to train and evaluate the model using statistical moments
or characteristics as features on a normal train/test split dataset
- `multilayer_perceptron.py`: Contains a script to train and evaluate a MLP that takes single cell features
as input and outputs if the cell is activated or not (obsolote as model does not work)
- `plot_frequency_ablation_second_freq.py`: Contains a script to plot the evaluation metrics changes
when removing a second frequency on top of frequency 5
- `plot_frequency_ablation_single_freq.py`: Contains a script to plot the evaluation metrics changes
when removing a single frequency from the feature set
- `plot_frequency_ablation_two_frequencies.py`: Contains a script to plot the evaluation metrics changes
when removing all possible two-frequency combinations from the feature set
- `plot_regularization_param.py`: Contains a script to plot the evaluation metrics changes
when changing the Ridge regularization parameter (obsolete as final model is ordinary linear regression)
- `plot_sampling_points.py`: Contains a script to plot the evaluation metrics changes when
changing the amount of sampling points that are used (obsolete as final model does not work with marginal distributions)
- `plot_sampling_range.py`: Contains a script to plot the evaluation metrics changes when
changing the sampling range (obsolete as final model does not work with marginal distributions)
- `requirements.txt`: Python virtual environment requirements
- `TODO.txt`: All todo notes during weekly meetings
- `visualization_umap.py`: Contains a script to visualize the different populations by running UMAP
on their features
- `viisualize_marginal_distributions.py`: Contains a script to estimate the marginal distribution
of each of the 17 features for a given population using kernel density estimation and plots the 
resulting distribution and the sampling points (obsolete as final model does not work with marginal distributions)

How to create paper figures:
- run data.py to generate a leave-one-out cross-validation dataset
- run model_moments_kfold.py with the frequencies to be removed as integer command line arguments to generate regression plots and confusion matrices in ./figures/moment_model/ and to create a results file in ./results/
- run model_moments_kfold.py with all possible one or two frequency combinations as command line arguments to create the result files for ablation studies
- manually adjust the reference Pearson, MAE and F1 score in plot_frequency_ablation_* files by using the result of model_moments_kfold.py when no frequency is omitted
- run all plot_frequency_ablation_* files to generate the frequency ablation plots
