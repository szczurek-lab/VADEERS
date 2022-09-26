# Code for "A generative recommender system with GMM prior for cancer drug generation and sensitivity prediction"

## Setup
- conda environment needed to execute the code can be created with a command: `conda env create --file conda.yml`

## Files
- models' implementations are at ./gmm-vae-compounds/models
- seeds for data splits are at ./gmm-vae-compounds/split_seeds.json

## Notebooks
- ./gmm-vae-compounds/train_gmm_vae_sens_model.ipynb - notebook for training sensitivity model with GMM VAE with guiding data as DVAE part
- ./gmm-vae-compounds/train_vanilla_vae_sens_model.ipynb - notebook for training sensitivity model with vanilla VAE as DVAE part
- ./gmm-vae-compounds/evaluate_predictive_performance.ipynb - notebook for evaluation of models' predictive performance
- ./gmm-vae-compounds/evaluate_generative_performance.ipynb - notebook for evaluation of models' generative performance
