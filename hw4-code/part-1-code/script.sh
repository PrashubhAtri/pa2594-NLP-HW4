#!/bin/bash

echo "---- Removing old packages ----"
pip uninstall -y transformers datasets tokenizers fsspec pyarrow numpy pandas scikit-learn evaluate tqdm nltk torch || true

echo "---- Removing HuggingFace cache ----"
rm -rf ~/.cache/huggingface/datasets
rm -rf ~/.cache/huggingface/modules

echo "---- Installing requested environment ----"
pip install --no-cache-dir -r requirements.txt

echo "---- Verifying versions ----"
python3 - << 'EOF'
import numpy, torch, datasets, transformers, sklearn, nltk, pyarrow, fsspec
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("datasets:", datasets.__version__)
print("transformers:", transformers.__version__)
print("sklearn:", sklearn.__version__)
print("nltk:", nltk.__version__)
print("pyarrow:", pyarrow.__version__)
print("fsspec:", fsspec.__version__)
EOF

echo "---- Done! ----"
