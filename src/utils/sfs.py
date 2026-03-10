import torch

SFS_CONFIG = {
    "fan":          {"stationary": True,  "n": 270},
    "pump":         {"stationary": True,  "n": 250},
    "slider":       {"stationary": False, "n": 7},
    "valve":        {"stationary": False, "n": 7},
    "ToyCar":       {"stationary": True,  "n": 230},
    "ToyConveyor":  {"stationary": True,  "n": 130},
}

def compute_sfs_mask(x: torch.Tensor, stationary: bool, n: int, temperature: float = 0.05) -> torch.Tensor:
    """
    Compute SFS mask for a given input tensor.

    x:          (1, n_mels, T) — single spectrogram, already normalized
    stationary: True  -> top-n most correlated frames are informative
                False -> top-n least correlated frames are informative
    n:          number of informative frames per the SFS-AEFS paper
    temperature: sharpness of the soft boundary around the n-th frame.
                 Smaller = closer to hard top-n selection.

    Returns:    (1, 1, T) soft mask in (0, 1)
    """
    frames = x.squeeze(0).T
    T_len = frames.shape[0]

    covar = torch.cov(frames.T)
    correl = covar.sum(dim=1) - covar.diagonal()

    if not stationary:
        correl = -correl

    # Find the n-th largest value as the soft threshold
    n_clamped = max(1, min(n, T_len))
    threshold = torch.topk(correl, n_clamped).values[-1]

    weights = torch.sigmoid((correl - threshold) / temperature)

    return weights.view(1, 1, -1)
