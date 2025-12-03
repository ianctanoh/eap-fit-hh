# Identifying multi-compartment Hodgkin-Huxley models with high-density extracellular voltage recordings

# eap-fit-hh

Code accompanying the paper: **[Identifying Multi-Compartment Hodgkin–Huxley Models with High-Density Extracellular Voltage Recordings](https://arxiv.org/abs/2506.20233)** by Ian C. Tanoh, Michael Deistler, Jakob Macke, and Scott Linderman.

This repository contains the code used in the paper to simulate multi-compartment Hodgkin–Huxley (HH) neuron models, compute extracellular voltages, and perform scalable state and parameter inference using variants of the Extended Kalman Filter (EKF). The goal is to estimate membrane voltages, ion channel gating variables, conductances, and the 3D position of a neuron using only extracellular recordings.

## What this code does
- Simulation of multi-compartment HH neurons using JAX and JAXLEY  
- Computation of extracellular voltages from transmembrane currents  
- State-space inference using:
  - Full EKF  
  - Diagonal EKF (diagEKF)  
  - Block-Diagonal EKF (BD-EKF)  
- Tools for fitting conductances, gating variables, and neuron geometry  
- Jupyter notebooks that reproduce all experiments from the paper (synthetic neurons, robustness tests, retinal ganglion cell, CA1 pyramidal neuron)

## Repository structure
- `eap_fit_hh/` — core library (models, filters, geometry, simulation, utils)  
- `experiments/` — Jupyter notebooks used in the paper

## Installation
Clone the repository: git clone https://github.com/ianctanoh/eap-fit-hh.git

Install dependencies: pip install -r requirements.txt


