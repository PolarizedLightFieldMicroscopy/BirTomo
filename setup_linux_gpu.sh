#!/usr/bin/env bash

conda create --name recon python=3.10 --yes
conda activate recon
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --yes
conda install tifffile tqdm matplotlib h5py -c conda-forge --yes
conda install plotly -c plotly --yes
pip install streamlit
conda install ipykernel pytest -c anaconda --yes

conda deactivate
