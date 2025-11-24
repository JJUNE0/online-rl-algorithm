# utils/wandb_utils.py
from __future__ import annotations
from typing import Any, Dict, Optional
from gymnasium.wrappers import RecordVideo
import os

try:
    import wandb  # optional
except Exception:
    wandb = None  # wandb 미설치/미사용 시에도 안전하게 동작

# ---- Public API -------------------------------------------------------------

def init_wandb(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    W&B 세션 초기화. 사용하지 않으면 no-op 핸들 반환.
    return: handle(dict) — 이후 모든 로그 함수에 넘겨서 사용.
    """
    enabled = bool(config.get("use_wandb", False)) and (wandb is not None)
    handle = {"enabled": enabled, "run": None}

    if not enabled:
        return handle

    project = config.get("wandb_project", "online_rl")
    group_name = f"{config['algorithm']['name']}-{config['env_id']}"
    run_name =  f"{group_name}-seed{config.get('random_seed', 0)}"

    if wandb.run is None:
        wandb.init(project=project, group=group_name, job_type="train_agent", name=run_name)
        try:
            # dict-like config만 업데이트
            if isinstance(config, dict):
                wandb.config.update(config, allow_val_change=True)
        except Exception:
            pass

    handle["run"] = wandb.run
    return handle


def log_train(handle: Dict[str, Any],
              total_steps: int,
              total_episodes: int,
              epi_return: Optional[float] = None,
              total_losses: Optional[Dict[str, float]] = None) -> None:
    """학습 스칼라 로깅: 스텝/에피소드/에피소드 리턴/손실들"""
    if not handle.get("enabled"):
        return
    data = {"global/steps": total_steps, "global/episodes": total_episodes}
    if epi_return is not None:
        data["train/episode_return"] = float(epi_return)
    if total_losses:
        for k, v in total_losses.items():
            try:
                data[f"loss/{k}"] = float(v)
            except Exception:
                pass
    wandb.log(data, step=total_steps)


def log_eval(handle: Dict[str, Any],
            total_steps: int,
            total_episodes: int,
            avg_return: float,
            max_return: float,
            min_return: float) -> None:
    """평가 성능 로깅"""
    if not handle.get("enabled"):
        return
    wandb.log({
        "eval/avg_return": float(avg_return),
        "eval/max_return": float(max_return),
        "eval/min_return": float(min_return),
        "global/steps": total_steps,
        "global/episodes": total_episodes,
    }, step=total_steps)


def log_speed(handle: Dict[str, Any],
              total_steps: int,
              stats: Dict[str, float]) -> None:
    """
    속도/지연 통계 로깅. 예:
    stats = {
      "speed/policy_ms_avg": ..., "speed/policy_ms_p95": ...,
      "speed/env_ms_avg": ...,    "speed/env_ms_p95": ...,
      "speed/loop_ms_avg": ...,   "speed/loop_ms_p95": ...,
      "speed/steps_per_sec": ...
    }
    """
    if not handle.get("enabled") or not stats:
        return
    wandb.log(stats, step=total_steps)


def log_checkpoint_artifact(handle: Dict[str, Any],
                            checkpoint_dir: str,
                            total_steps: int) -> None:
    """체크포인트 디렉터리를 W&B Artifact로 업로드(옵션)."""
    if not handle.get("enabled"):
        return
    try:
        if not os.path.isdir(checkpoint_dir):
            return
        art = wandb.Artifact(f"ckpt_step_{total_steps}", type="model")
        art.add_dir(checkpoint_dir)
        wandb.log_artifact(art)
    except Exception as e:
        print(f"[wandb] artifact upload failed: {e}")


def finish(handle: Dict[str, Any]) -> None:
    """W&B 세션 종료 (안 쓰면 no-op)."""
    if not handle.get("enabled"):
        return
    try:
        if handle.get("run") is not None:
            wandb.finish()
    except Exception:
        pass
