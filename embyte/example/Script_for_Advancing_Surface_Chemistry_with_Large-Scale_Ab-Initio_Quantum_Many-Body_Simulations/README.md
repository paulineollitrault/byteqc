# Guide

This guide provides a comprehensive walkthrough starting from environment configuration to properly run all script files in this directory for the scriptes of the paper "Advancing Surface Chemistry with Large-Scale Ab-Initio Quantum Many-Body Simulations". Readers are encouraged to patiently follow this guide step-by-step to complete all required setups.

## System information
The package of ByteQC is running on Linux type system, which we used for coding and testing with the information as following.
```
Distributor ID: Debian
Description:    Debian GNU/Linux 12
Release:        12
```
The code for GPU part is running on NVIDIA GPUs with the hardware information as following.
```
GPU:            NVIDIA A100-SXM4-80GB
CPU:            Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz, 14 cores
Memory:         245 GB
```
The code for CPU part, like some Hartree-Fock calculation. is running on the hardware with the information as following.
```
CPU:            Intel(R) Xeon(R) Platinum 8336C CPU @ 2.30GHz, 96 cores
Memory:         980 GB
```
**Please Note:**

1. The configurations listed above are highly recommended. If your system does not meet these specifications, the program can still run, but you may encounter **out-of-memory (OOM)** errors in large systems. When GPU VRAM is exhausted, `cupy` will display relevant error. However, if CPU memory is depleted, the program will be terminated abruptly without any reminder.
2. If you have multiple GPUs available, you can proportionally scale the allocated CPU cores and memory size based on the ratios mentioned above to achieve optimal performance.

## Environment configuration
We highly recommand user to set the environment with the help of `conda`. Therefore, we will from install `miniconda` as following command.
```
#! /bin/bash

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
```
