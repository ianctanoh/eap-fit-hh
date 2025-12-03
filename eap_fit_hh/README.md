# Identifying multi-compartment Hodgkin-Huxley models with high-density extracellular voltage recordings

# eap-fit-hh

Code accompanying the paper:

**[Identifying Multi-Compartment Hodgkin–Huxley Models with High-Density Extracellular Voltage Recordings](https://arxiv.org/abs/2506.20233)**  
Ian C. Tanoh, Michael Deistler, Jakob Macke, Scott Linderman

This repository contains implementations of the inference methods in the paper, including simulation, extracellular voltage prediction, and scalable variants of the Extended Kalman Filter (EKF) for fitting multi-compartment Hodgkin–Huxley (HH) neuron models.

---

## What this code does

This project fits detailed HH neuron models **directly to extracellular voltage recordings**, without needing intracellular recordings.

We combine:

- **JAX + JAXLEY** for differentiable simulation of multi-compartment HH neurons  
- **Forward modeling of extracellular potentials** from transmembrane currents  
- **State-space inference** to estimate:
  - Membrane voltages
  - Ion channel gating variables
  - Maximum conductances
  - Spatial position of the neuron relative to the probe

To scale inference to large models (hundreds–thousands of states), we implement:

- **Full EKF** — accurate but expensive  
- **Diagonal EKF (diagEKF)** — fast, assumes independent state uncertainty  
- **Block-Diagonal EKF (BD-EKF)** — dense voltage covariance + diagonal gating covariance; fast and accurate for complex morphologies

The notebooks in `experiments/` reproduce the main results from the paper, including synthetic validation, robustness experiments, and applications to real morphologies (retinal ganglion cell, CA1 pyramidal cell).

---

## Repository structure

```text
eap_fit_hh/
│
├── eap_fit_hh/        # Core library: models, filters, geometry, utils
└── experiments/       # Jupyter notebooks for reproducing the paper's results
