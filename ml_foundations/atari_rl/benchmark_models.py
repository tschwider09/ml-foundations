from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from .environment import create_atari_env
from .evaluation import evaluate_greedy_policy
from .saved_models import (
    BENCHMARK_RESULTS_PATH,
    BEST_PLAYBACK_MODEL,
    DEFAULT_PLAYBACK_MODEL,
    MODELS_DIR,
    SavedModelArtifact,
    discover_saved_models,
    load_saved_model_artifact,
)


def _compact_error(exc: Exception, *, max_chars: int = 240) -> str:
    line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    if len(line) <= max_chars:
        return line
    return line[: max_chars - 3] + "..."


@dataclass
class BenchmarkResult:
    artifact_path: str
    artifact_kind: str
    episodes: int
    scores: list[float]
    mean_reward: float
    max_reward: float
    min_reward: float
    promoted_model_path: str | None = None


def evaluate_artifact(
    artifact: SavedModelArtifact,
    *,
    episodes: int,
    env_name: str,
    max_steps_per_episode: int | None,
    seed_base: int | None,
) -> BenchmarkResult:
    """Evaluate one saved-model artifact in a fresh headless Atari environment."""
    env = create_atari_env(env_name, render_mode=None, frame_skip=4, stack_size=4)
    try:
        model = load_saved_model_artifact(
            artifact,
            input_shape=env.observation_space.shape,
            n_actions=env.action_space.n,
        )
        scores = evaluate_greedy_policy(
            model,
            env,
            episodes=episodes,
            sleep_s=0.0,
            max_steps_per_episode=max_steps_per_episode,
            seed_base=seed_base,
        )
    finally:
        env.close()

    return BenchmarkResult(
        artifact_path=str(artifact.path),
        artifact_kind=artifact.kind,
        episodes=episodes,
        scores=[float(x) for x in scores],
        mean_reward=float(np.mean(scores)),
        max_reward=float(np.max(scores)),
        min_reward=float(np.min(scores)),
    )


def _copy_if_needed(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)


def promote_best_model(
    artifact: SavedModelArtifact,
    *,
    env_name: str,
) -> Path:
    """
    Promote the best artifact into canonical playback locations.
    Returns the canonical best model path.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if artifact.kind == "keras":
        _copy_if_needed(artifact.path, BEST_PLAYBACK_MODEL)
        _copy_if_needed(artifact.path, DEFAULT_PLAYBACK_MODEL)
        return BEST_PLAYBACK_MODEL

    env = create_atari_env(env_name, render_mode=None, frame_skip=4, stack_size=4)
    try:
        model = load_saved_model_artifact(
            artifact,
            input_shape=env.observation_space.shape,
            n_actions=env.action_space.n,
        )
    finally:
        env.close()

    model.save(BEST_PLAYBACK_MODEL)
    model.save(DEFAULT_PLAYBACK_MODEL)
    return BEST_PLAYBACK_MODEL


def benchmark_saved_models(
    *,
    models_dir: Path = MODELS_DIR,
    episodes: int = 2,
    env_name: str = "ALE/SpaceInvaders-v5",
    include_weights: bool = True,
    promote_best: bool = True,
    results_path: Path = BENCHMARK_RESULTS_PATH,
    max_steps_per_episode: int | None = None,
    seed_base: int | None = 123,
):
    """Evaluate all saved models, rank them, and optionally promote top model."""
    artifacts = discover_saved_models(models_dir=models_dir, include_weights=include_weights)
    if not artifacts:
        raise FileNotFoundError(f"No models found in {Path(models_dir).resolve()}")

    ranked: list[BenchmarkResult] = []
    failed: list[dict[str, str]] = []
    artifact_by_path = {str(item.path): item for item in artifacts}

    for artifact in artifacts:
        print(f"Evaluating {artifact.path.name} ({artifact.kind}) ...")
        try:
            result = evaluate_artifact(
                artifact,
                episodes=episodes,
                env_name=env_name,
                max_steps_per_episode=max_steps_per_episode,
                seed_base=seed_base,
            )
            ranked.append(result)
            print(
                f"  mean={result.mean_reward:.2f} "
                f"max={result.max_reward:.2f} min={result.min_reward:.2f}"
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            compact = _compact_error(exc)
            failed.append(
                {
                    "artifact_path": str(artifact.path),
                    "artifact_kind": artifact.kind,
                    "error": compact,
                }
            )
            print(f"  failed: {compact}")

    ranked.sort(
        key=lambda item: (item.mean_reward, item.max_reward, -item.min_reward),
        reverse=True,
    )

    best_model_path: str | None = None
    if promote_best and ranked:
        best_artifact = artifact_by_path[ranked[0].artifact_path]
        promoted = promote_best_model(best_artifact, env_name=env_name)
        ranked[0].promoted_model_path = str(promoted)
        best_model_path = str(promoted)

    payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "env_name": env_name,
        "episodes_per_model": episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "seed_base": seed_base,
        "models_dir": str(Path(models_dir).resolve()),
        "include_weights": include_weights,
        "best_model_path": best_model_path,
        "ranked_results": [asdict(item) for item in ranked],
        "failed_models": failed,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2))

    return payload


def _print_leaderboard(payload: dict):
    ranked = payload["ranked_results"]
    if not ranked:
        print("No models successfully evaluated.")
        return

    print("\nModel leaderboard:")
    for idx, item in enumerate(ranked, start=1):
        name = Path(item["artifact_path"]).name
        print(
            f"{idx:>2}. {name:<35} "
            f"mean={item['mean_reward']:>7.2f} "
            f"max={item['max_reward']:>7.2f} "
            f"min={item['min_reward']:>7.2f}"
        )

    if payload.get("best_model_path"):
        print(f"\nBest model promoted to: {payload['best_model_path']}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark saved Space Invaders model artifacts.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Episodes per model for ranking.",
    )
    parser.add_argument(
        "--env-name",
        default="ALE/SpaceInvaders-v5",
        help="Gymnasium environment id.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory containing .keras and/or .weights.h5 files.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=BENCHMARK_RESULTS_PATH,
        help="Where to write benchmark ranking JSON.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1500,
        help="Optional step cap per episode to keep benchmarking bounded.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base seed used for env.reset so every model is evaluated on the same starts.",
    )
    parser.add_argument(
        "--keras-only",
        action="store_true",
        help="Skip .weights.h5 files and only benchmark .keras files.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Do not copy/save the best model to canonical playback filenames.",
    )
    args = parser.parse_args()

    payload = benchmark_saved_models(
        models_dir=args.models_dir,
        episodes=args.episodes,
        env_name=args.env_name,
        include_weights=not args.keras_only,
        promote_best=not args.no_promote,
        results_path=args.results_json,
        max_steps_per_episode=args.max_steps if args.max_steps > 0 else None,
        seed_base=args.seed,
    )
    _print_leaderboard(payload)
    print(f"\nBenchmark JSON written to: {args.results_json.resolve()}")


if __name__ == "__main__":
    main()
