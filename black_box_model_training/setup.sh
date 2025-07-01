
# 1. Create fresh environment
conda create -n tm_env python=3.11 -y
conda activate tm_env

# 2. Core libs
pip install torch-molecule deepchem scikit-learn torch-scatter transformers tensorflow dgl pytorch-lightning jax haiku 