# Install python 3.6 and set up virtual environment
conda create -y --name loho python=3.6
conda activate loho 

# Install packages
pip install torchsummary
conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -y -c conda-forge tqdm
conda install -y -c conda-forge python-lmdb
conda install -y -c conda-forge mlflow
conda install -y -c anaconda scikit-image 
conda install -y -c conda-forge ipdb
conda install -y -c anaconda scipy
conda install -y -c conda-forge opencv
conda install -y -c conda-forge dominate
conda install -y -c anaconda dill
conda install -y -c conda-forge matplotlib
conda install -y -c conda-forge tensorboardx
conda install -y -c conda-forge tensorboard
conda install -y -c anaconda boto3
conda install -y -c conda-forge python-dotenv 
