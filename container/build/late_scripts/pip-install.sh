#!/bin/bash
echo "Ensure this is running as user if issues arise..."
whoami
# the environment will go into user home b/c we are running as user.
/opt/conda/bin/conda create -n myenv python=3.10
# use conda for faiss install.
/opt/conda/bin/conda install -n myenv -c pytorch -c nvidia faiss-gpu=1.8.0 -y
~/.conda/envs/myenv/bin/python3 -m pip install --upgrade pip
~/.conda/envs/myenv/bin/python3 -m pip install -r /cfg/requirements.txt
