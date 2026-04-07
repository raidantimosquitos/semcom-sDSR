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

## BER-calibrated bitflip baselines (OPUS/JPEG)

To avoid extremely slow per-clip LDPC decoding for OPUS/JPEG sweeps, you can:

- **Calibrate** a post-FEC residual BER vs SNR curve once for a fixed LDPC(≈1/2)+BPSK config:

```bash
python3 scripts/calibrate_ldpc_bpsk_ber.py --out results/ber_curve.csv
```

- **Use** that BER curve to inject i.i.d. bit flips into codec bytes (optionally protecting headers):

```bash
python3 -m scripts.evaluate_awgn_jpeg \
  --stage1_ckpt <...> --stage2_ckpt <...> --data_path <...> --machine_type <...> \
  --use_channel --ber_curve results/ber_curve.csv --protect_bytes 64

python3 -m scripts.evaluate_awgn_opus \
  --stage1_ckpt <...> --stage2_ckpt <...> --data_path <...> --machine_type <...> \
  --use_channel --ber_curve results/ber_curve.csv --protect_bytes 128
```

Notes:
- The **BER(SNR) curve depends on PHY/FEC settings** (LDPC, iterations, modulation), not on OPUS bitrate or JPEG Q.
- Codec settings mainly affect **fragility** (decode success rate and reconstruction quality) at a given residual BER.