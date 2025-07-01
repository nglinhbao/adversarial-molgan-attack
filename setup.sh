#!/bin/bash

# Initialize JT-VAE submodule like mol-cycle-gan does
echo "Initializing JT-VAE submodule..."
git submodule add https://github.com/hello-maker/JunctionTreeVAE jtvae
git submodule update --init --recursive

# Create __init__.py for jtvae module like mol-cycle-gan
cat > jtvae/__init__.py << INNER_EOF
from .jtnn.mol_tree import Vocab, MolTree
from .jtnn.jtnn_vae import JTNNVAE
INNER_EOF

cat > jtvae/jtnn/__init__.py << INNER_EOF
from .mol_tree import Vocab, MolTree
from .jtnn_vae import JTNNVAE
from .jtprop_vae import JTPropVAE
from .mpn import MPN, mol2graph
from .nnutils import create_var
from .datautils import MoleculeDataset, PropDataset
from .chemutils import decode_stereo
INNER_EOF

sed -i 's/from chemutils/from .chemutils/g' jtvae/jtnn/mol_tree.py

sed -i 's/from mol_tree/from .mol_tree/g' jtvae/jtnn/jtnn_vae.py
sed -i 's/from mol_tree/from .nnutils/g' jtvae/jtnn/jtnn_vae.py

# Install other dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models
mkdir -p data
mkdir -p results

echo "Setup complete!"
