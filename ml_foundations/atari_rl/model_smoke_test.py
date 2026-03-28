from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .environment import create_atari_env
from .saved_models import MODELS_DIR, discover_saved_models, load_saved_model_artifact


def _compact_error(exc: Exception, *, max_chars: int = 240) -> str:
    line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    if len(line) <= max_chars:
        return line
    return line[: max_chars - 3] + "..."


def run_model_smoke_test(
    *,
    models_dir: Path = MODELS_DIR,
    env_name: str = "ALE/SpaceInvaders-v5",
    include_weights: bool = True,
    strict: bool = False,
):
    """Verify each saved model loads and produces valid Q-values."""
    artifacts = discover_saved_models(models_dir=models_dir, include_weights=include_weights)
    if not artifacts:
        raise FileNotFoundError(f"No model artifacts found in {Path(models_dir).resolve()}")

    env = create_atari_env(env_name, render_mode=None, frame_skip=4, stack_size=4)
    try:
        state, _ = env.reset()
        input_shape = env.observation_space.shape
        n_actions = env.action_space.n

        failures: list[str] = []
        passes = 0

        for artifact in artifacts:
            try:
                model = load_saved_model_artifact(
                    artifact,
                    input_shape=input_shape,
                    n_actions=n_actions,
                )
                q_values = model(state[np.newaxis, :], training=False).numpy()
                if q_values.shape != (1, n_actions):
                    raise ValueError(
                        f"Unexpected output shape {q_values.shape}, expected (1, {n_actions})"
                    )
                passes += 1
                print(f"[PASS] {artifact.path.name} ({artifact.kind})")
            except Exception as exc:  # pragma: no cover - diagnostic path
                compact = _compact_error(exc)
                failures.append(f"{artifact.path.name}: {compact}")
                print(f"[FAIL] {artifact.path.name} ({artifact.kind}): {compact}")
    finally:
        env.close()

    print(f"Smoke test summary: {passes} passed, {len(failures)} failed")
    if failures and strict:
        raise RuntimeError("Model smoke test failures:\n" + "\n".join(failures))
    if failures and not strict:
        print("Non-strict mode: failed artifacts were reported but did not fail the run.")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test Space Invaders saved models.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory containing .keras and/or .weights.h5 artifacts.",
    )
    parser.add_argument(
        "--env-name",
        default="ALE/SpaceInvaders-v5",
        help="Gymnasium environment id.",
    )
    parser.add_argument(
        "--keras-only",
        action="store_true",
        help="Only test .keras files, skip .weights.h5 artifacts.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit code if any artifact fails to load/infer.",
    )
    args = parser.parse_args()

    run_model_smoke_test(
        models_dir=args.models_dir,
        env_name=args.env_name,
        include_weights=not args.keras_only,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
