conda create -n genmol python==3.10
conda activate genmol
pip install -r env/requirements.txt
pip install -e .
pip install scikit-learn==1.2.2     # required to run hit generation (gsk3b, jnk3)
