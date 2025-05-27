from dataclasses import dataclass
from typing import Set

@dataclass
class ExperimentConfig:
    input_dir: str
    output_dir: str
    output_file: str
    masks_json: str
    slow_models: Set[str]
    slow_model_n: int
    fast_model_n: int
    leave_free_cores: int = 2
    csv_out: bool = True