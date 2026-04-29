from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    data_path: Path
    output_dir: Path = Path("artifacts")
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "Class"
    drop_columns: list[str] = field(default_factory=list)
    anomaly_contamination: float | str = "auto"
    pca_sample_size: int = 5000
    shap_sample_size: int = 2000
    feature_importance_top_k: int = 20
