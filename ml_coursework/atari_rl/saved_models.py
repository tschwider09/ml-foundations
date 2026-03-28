from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import tensorflow as tf

from .modeling import build_dqn_cnn

MODEL_PATH_ENV_VAR = "SPACE_INVADERS_MODEL_PATH"
MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_PLAYBACK_MODEL = MODELS_DIR / "dqn_space_invaders.keras"
BEST_PLAYBACK_MODEL = MODELS_DIR / "dqn_space_invaders_best.keras"
BENCHMARK_RESULTS_PATH = MODELS_DIR / "model_benchmark_results.json"


@dataclass(frozen=True)
class SavedModelArtifact:
    path: Path
    kind: str  # "keras" or "weights"


def discover_saved_models(
    models_dir: Path = MODELS_DIR,
    *,
    include_weights: bool = True,
) -> list[SavedModelArtifact]:
    """Discover saved model artifacts in a consistent sort order."""
    models_dir = Path(models_dir).expanduser().resolve()
    artifacts: list[SavedModelArtifact] = []

    for path in sorted(models_dir.glob("*.keras")):
        artifacts.append(SavedModelArtifact(path=path, kind="keras"))

    if include_weights:
        for path in sorted(models_dir.glob("*.weights.h5")):
            artifacts.append(SavedModelArtifact(path=path, kind="weights"))

    return artifacts


def load_saved_model_artifact(
    artifact: SavedModelArtifact,
    *,
    input_shape: tuple[int, int, int],
    n_actions: int,
):
    """Load either a full .keras model or .weights.h5 into the shared DQN architecture."""
    if artifact.kind == "keras":
        return tf.keras.models.load_model(artifact.path)

    if artifact.kind == "weights":
        model = build_dqn_cnn(input_shape, n_actions)
        model.load_weights(artifact.path)
        return model

    raise ValueError(f"Unsupported model artifact kind: {artifact.kind}")


def resolve_benchmark_best_model(
    results_path: Path = BENCHMARK_RESULTS_PATH,
) -> Path | None:
    """Return the promoted best model path from benchmark results when available."""
    if not results_path.exists():
        return None

    try:
        payload = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    best_path_value = payload.get("best_model_path")
    if isinstance(best_path_value, str):
        path = Path(best_path_value).expanduser().resolve()
        if path.exists():
            return path

    ranked = payload.get("ranked_results")
    if isinstance(ranked, list) and ranked:
        top = ranked[0]
        if isinstance(top, dict):
            promoted = top.get("promoted_model_path")
            if isinstance(promoted, str):
                promoted_path = Path(promoted).expanduser().resolve()
                if promoted_path.exists():
                    return promoted_path
            artifact = top.get("artifact_path")
            if isinstance(artifact, str):
                artifact_path = Path(artifact).expanduser().resolve()
                if artifact_path.exists() and artifact_path.suffix == ".keras":
                    return artifact_path

    return None


def resolve_playback_model_path(
    cli_model_path: str | None = None,
) -> tuple[Path, str]:
    """
    Resolve playback model path in priority order:
    CLI arg -> env var -> benchmark-selected model -> explicit best path
    -> default path -> newest available .keras file.
    """
    if cli_model_path:
        return Path(cli_model_path).expanduser().resolve(), "--model"

    env_model_path = os.getenv(MODEL_PATH_ENV_VAR)
    if env_model_path:
        return Path(env_model_path).expanduser().resolve(), MODEL_PATH_ENV_VAR

    benchmark_best = resolve_benchmark_best_model()
    if benchmark_best is not None:
        return benchmark_best, "benchmark_results"

    if BEST_PLAYBACK_MODEL.exists():
        return BEST_PLAYBACK_MODEL, "best_model_fallback"

    if DEFAULT_PLAYBACK_MODEL.exists():
        return DEFAULT_PLAYBACK_MODEL, "default_model_path"

    keras_candidates = discover_saved_models(include_weights=False)
    if keras_candidates:
        newest = max(keras_candidates, key=lambda item: item.path.stat().st_mtime)
        return newest.path, "newest_keras_candidate"

    return DEFAULT_PLAYBACK_MODEL, "default_model_path"
