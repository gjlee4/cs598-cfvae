# CS598-CFVAE: Counterfactual Variational Autoencoders

This repository contains an implementation of Counterfactual Variational Autoencoders (CF-VAE) for healthcare applications, specifically focusing on predicting patient interventions.

## Overview

The CF-VAE model combines a Variational Autoencoder (VAE) with a multi-task prediction network to generate counterfactual representations of patient data. This approach allows for both prediction of interventions and ranking of patients by severity, providing interpretable insights for clinical decision-making.

## Dataset

The implementation uses the MIMIC-III dataset (not included in the repository) to:

- Extract ICU stays of patients ≥18 years old with length of stay ≥54 hours
- Process vital signs and lab measurements as time series features
- Define interventions based on vasopressor administration in a future time window

## Files in the Repository

- `model.py`: Contains the implementation of the MultiTaskMLPModel and CFVAE architectures
- `replicating_cfvae.ipynb`: Main notebook demonstrating the implementation and evaluation of the CF-VAE model on MIMIC-III data
- `cfvae_on_mnist_extension.ipynb`: Extension of the CF-VAE approach to the MNIST dataset
- `cfvae_best.pt`: Pre-trained CF-VAE model weights
- `predictor_multitask.pt`: Pre-trained multi-task prediction model weights

## Installation

```bash
pip install torch numpy pandas matplotlib scikit-learn
