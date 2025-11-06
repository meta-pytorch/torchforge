from dataclasses import dataclass


@dataclass
class DatasetConfig:
    source: str
    split: str = "train"
