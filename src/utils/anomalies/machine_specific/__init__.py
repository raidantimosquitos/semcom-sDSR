"""
Machine-specific anomaly mask generators for DCASE2020 Task 2 machine types.

Each module exposes a single callable class with the interface:
    strategy(batch_size, device) -> (B, 1, n_mels, T) binary torch.Tensor

Available strategies (keyed by machine_type string, lowercase):
    fan          -> FanAnomalyStrategy
    pump         -> PumpAnomalyStrategy
    slider       -> SliderAnomalyStrategy
    toycar       -> ToyCarAnomalyStrategy
    toyconveyor  -> ToyConveyorAnomalyStrategy
    valve        -> ValveAnomalyStrategy
"""

from .fan import FanAnomalyStrategy
from .pump import PumpAnomalyStrategy
from .slider import SliderAnomalyStrategy
from .toycar import ToyCarAnomalyStrategy
from .toyconveyor import ToyConveyorAnomalyStrategy
from .valve import ValveAnomalyStrategy

STRATEGY_MAP: dict[str, type] = {
    "fan": FanAnomalyStrategy,
    "pump": PumpAnomalyStrategy,
    "slider": SliderAnomalyStrategy,
    "toycar": ToyCarAnomalyStrategy,
    "toyconveyor": ToyConveyorAnomalyStrategy,
    "valve": ValveAnomalyStrategy,
}

__all__ = [
    "FanAnomalyStrategy",
    "PumpAnomalyStrategy",
    "SliderAnomalyStrategy",
    "ToyCarAnomalyStrategy",
    "ToyConveyorAnomalyStrategy",
    "ValveAnomalyStrategy",
    "STRATEGY_MAP",
]