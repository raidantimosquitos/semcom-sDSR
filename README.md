FYI

## Install conda 

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

## Clone the repository and install the Python environment
```bash
git clone https://github.com/raidantimosquitos/semcom-sDSR.git
cd semcom-sDSR
conda env create -f environment.yml
conda run -n sDSR pip install --no-build-isolation pyldpc
```
(`pyldpc` is not on conda; its sdist build needs the env’s NumPy, so build isolation is disabled. Alternatively run `bash scripts/bootstrap_conda_env.sh`.)