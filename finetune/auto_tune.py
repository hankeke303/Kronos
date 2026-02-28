#!/usr/bin/env python3
"""
Auto hyperparameter tuning runner for Kronos.
- Launches tokenizer or predictor training with early stopping.
- After predictor training, runs qlib_test and kronostest backtests.
- Saves config snapshots and logs results to a JSONL history file.
- Queries an LLM (OpenAI or vLLM HTTP) for the next hyperparameter suggestion.

Stop the loop with Ctrl-C.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent))
from config import Config  # noqa: E402


def _ts_id(target: str) -> str:
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{target}_{now}"


def _ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _append_history(path: Path, row: Dict[str, Any]) -> None:
    _ensure_dirs(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _snapshot_config(config_path: Path, run_id: str) -> Path:
    dst = config_path.parent / "configs" / f"config_{run_id}.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, dst)
    return dst


def _update_config_file(config_path: Path, updates: Dict[str, Any], new_run_id: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    updates = dict(updates)
    updates["run_id"] = new_run_id

    def _fmt(val: Any) -> str:
        if isinstance(val, str):
            return f"\"{val}\""
        return repr(val)

    for key, val in updates.items():
        replaced = False
        needle = f"self.{key} ="
        for idx, line in enumerate(lines):
            # Skip commented lines to avoid modifying disabled assignments
            stripped = line.lstrip()
            if stripped.startswith('#'):
                continue
            if needle in stripped:
                indent = line.split("self.")[0]
                lines[idx] = f"{indent}self.{key} = {_fmt(val)}"
                replaced = True
                break
        if not replaced:
            print(f"[warn] Key {key} not found in config.py; skipped")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _ensure_run_id_in_file(config_path: Path, run_id: str) -> None:
    """Ensure config.py contains a self.run_id assignment inside Config.__init__."""
    text = config_path.read_text(encoding="utf-8")
    if "self.run_id" in text:
        return
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "def __init__(self):" in line:
            # Insert after __init__ declaration with 8-space indent
            insert_line = "        self.run_id = \"" + run_id + "\""
            lines.insert(idx + 1, insert_line)
            config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    # Fallback: append at end (not ideal but prevents missing run_id)
    lines.append("\n    def __init__(self):\n        self.run_id = \"" + run_id + "\"")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_run_id_from_file(config_path: Path) -> Optional[str]:
    """Parse self.run_id value from config.py; returns None if not found."""
    try:
        text = config_path.read_text(encoding="utf-8")
    except Exception:
        return None
    # Look for a line: self.run_id = "..." or '...'
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if "self.run_id" in stripped and "=" in stripped:
            try:
                _, rhs = stripped.split("=", 1)
                rhs = rhs.strip()
                if rhs.startswith('"') and '"' in rhs[1:]:
                    return rhs.split('"')[1]
                if rhs.startswith("'") and "'" in rhs[1:]:
                    return rhs.split("'")[1]
            except Exception:
                pass
    return None


def _launch(cmd: List[str], workdir: Path | None = None, stream: bool = True) -> tuple[bool, str, str]:
    print(f"[cmd] {' '.join(cmd)}")
    if stream:
        # Stream stdout+stderr live to console while capturing combined output
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        combined: List[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                combined.append(line)
            proc.wait()
            success = proc.returncode == 0
            return success, "".join(combined), ""
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
    else:
        try:
            proc = subprocess.run(cmd, cwd=workdir, check=True, capture_output=True, text=True)
            return True, proc.stdout or "", proc.stderr or ""
        except subprocess.CalledProcessError as e:
            return False, e.stdout or "", e.stderr or ""


def _summary_from_save(save_dir: Path) -> Dict[str, Any]:
    summary_path = save_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


class LLMClient:
    def __init__(self, provider: str, model: str, timeout: int = 30):
        self.provider = provider
        self.model = model
        self.timeout = timeout

    def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=messages,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content.strip()

    def _call_vllm(self, prompt: str) -> str:
        import requests
        base_url = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1/completions")
        resp = requests.post(
            base_url,
            timeout=self.timeout,
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 4096,
            },
        )
        resp.raise_for_status()
        return resp.json().get("choices", [{}])[0].get("text", "")

    def _extract_code(self, text: str) -> str:
        if "```" in text:
            parts = text.split("```")
            for i in range(0, len(parts) - 1, 2):
                fence_lang = parts[i].strip().lower()
                code = parts[i + 1]
                if fence_lang.endswith("python") or fence_lang == "":
                    return code.strip()
        return text.strip()

    def generate_config(self, current_config_text: str, target: str, detailed_context: str) -> str:
        system = "You are a helpful AI that outputs valid Python files only."
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user = (
            "请根据下面现有的 config.py，生成一个新的完整 config.py 文件，用于训练 '" + target + "'。\n"
            "背景任务与架构说明（节选）：\n" + detailed_context + "\n"
            "当前日期时间：" + now + "（请结合到 self.run_id 中，便于区分）。\n"
            "要求：\n- 只输出完整 Python 代码文件。\n- 保留必要字段与结构；包含唯一的 self.run_id（带时间戳/目标等信息）。\n- 保留或设置早停参数 early_stop_patience=10。\n- 其他超参可按合理原则调整。\n\n"
            "现有 config.py 内容如下：\n\n" + current_config_text
        )
        try:
            if self.provider == "openai":
                content = self._call_openai([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
            elif self.provider == "vllm":
                content = self._call_vllm(user)
            else:
                return current_config_text
        except Exception as exc:
            print(f"[warn] LLM generate_config failed: {exc}")
            return current_config_text
        return self._extract_code(content)

    def fix_config_on_error(self, current_config_text: str, target: str, error_text: str, detailed_context: str) -> str:
        system = "You are a helpful AI that outputs valid Python files only."
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user = (
            "新的 config.py 刚刚运行报错了。请根据错误信息修正整个 config.py 并输出完整代码：\n\n"
            + error_text + "\n\n原始（当前）config.py：\n\n" + current_config_text + "\n\n目标：" + target + "\n"
            + "背景任务与架构说明（节选）：\n" + detailed_context + "\n"
            + "当前日期时间：" + now + "（如果需要请更新 self.run_id 以便区分）。\n"
        )
        try:
            if self.provider == "openai":
                content = self._call_openai([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
            elif self.provider == "vllm":
                content = self._call_vllm(user)
            else:
                return current_config_text
        except Exception as exc:
            print(f"[warn] LLM fix_config_on_error failed: {exc}")
            return current_config_text
        return self._extract_code(content)

    def warmup(self, base_cfg: Config) -> bool:
        try:
            content = self.generate_config(
                "class Config:\n    def __init__(self):\n        self.run_id = 'warmup'\n",
                target="predictor",
                detailed_context="(warmup)"
            )
            ok = isinstance(content, str) and len(content) > 10
            print("[llm] Warmup:", "ok" if ok else "no response")
            return ok
        except Exception as exc:
            print(f"[llm] Warmup error: {exc}")
            return False


def _run_pipeline(args: argparse.Namespace) -> None:
    base_cfg = Config()
    history_path = Path(args.history_path)
    history = _load_history(history_path)
    llm = LLMClient(args.llm_provider, args.llm_model)
    finetune_dir = Path(__file__).resolve().parent
    repo_root = finetune_dir.parent
    config_path = finetune_dir / "config.py"

    # Snapshot current config before the first modification
    _snapshot_config(config_path, Config().run_id)

    # Early check: test LLM API availability before running loop
    _ = llm.warmup(base_cfg)

    candidate_updates: Dict[str, Any] = {}
    # Prepare detailed context from README and model definition for prompt enrichment
    try:
        readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    except Exception:
        readme_text = ""
    try:
        arch_text = (repo_root / "model" / "kronos.py").read_text(encoding="utf-8")
    except Exception:
        arch_text = ""
    # Trim to reasonable size to avoid exceeding token limits
    detailed_context = (readme_text[:3000] + "\n\n" + arch_text[:3000]) if (readme_text or arch_text) else "(无额外上下文)"

    # full-config generation replaces notes/updates; keep for history compatibility
    run_idx = 0

    while args.max_runs is None or run_idx < args.max_runs:
        run_id = _ts_id(args.target)
        # Read current config
        current_text = config_path.read_text(encoding="utf-8")
        # Ask LLM to produce an entire new config.py
        new_text = llm.generate_config(current_text, args.target, detailed_context)
        # Write new config
        config_path.write_text(new_text, encoding="utf-8")
        # Ensure run_id exists and is set to our unique id
        # If LLM didn't include run_id, insert a fallback and then try to parse the final value
        _ensure_run_id_in_file(config_path, run_id)
        # Try to extract the LLM-provided run_id for consistent folder naming
        llm_run_id = _extract_run_id_from_file(config_path) or run_id

        train_script = "train_predictor.py" if args.target == "predictor" else "train_tokenizer.py"
        train_cmd = [
            "torchrun", "--standalone", f"--nproc_per_node={args.nproc}",
            str(Path(__file__).parent / train_script),
            "--mode", "finetune",
            "--init", args.init,
        ]
        # Run from repo root so relative data paths (e.g., ./data/processed_datasets_110) resolve correctly.
        ok, out, err = _launch(train_cmd, workdir=repo_root)
        if not ok:
            # Fix config using error feedback and retry once
            current_text = config_path.read_text(encoding="utf-8")
            fixed_text = llm.fix_config_on_error(current_text, args.target, (out or "") + "\n" + (err or ""), detailed_context)
            config_path.write_text(fixed_text, encoding="utf-8")
            _ensure_run_id_in_file(config_path, run_id)
            llm_run_id = _extract_run_id_from_file(config_path) or run_id
            ok, out, err = _launch(train_cmd, workdir=repo_root)
        if not ok:
            print("[train] Failed after retry. Skipping this run.")
            snapshot_path = _snapshot_config(config_path, run_id)
            row = {
                "ts": datetime.utcnow().isoformat(),
                "run_id": run_id,
                "target": args.target,
                "init": args.init,
                "error": (out or "") + "\n" + (err or ""),
                "stage": "train",
                "config_snapshot": str(snapshot_path),
            }
            _append_history(history_path, row)
            history.append(row)
            run_idx += 1
            print(f"[info] Completed run {run_id} with failure. Waiting {args.cooldown}s before next.")
            time.sleep(args.cooldown)
            continue

        save_dir = Path(base_cfg.save_path) / (f"finetune_predictor_demo_{llm_run_id}" if args.target == "predictor" else f"finetune_tokenizer_demo_{llm_run_id}")
        summary = _summary_from_save(save_dir)
        final_result = summary.get("final_result", {}) if summary else {}

        backtest_summary: Dict[str, Any] = {}
        if args.target == "predictor":
            qlib_cmd = [
                sys.executable,
                str(finetune_dir / "qlib_test.py"),
                "--device", args.device,
            ]
            ok_bt, out_bt, err_bt = _launch(qlib_cmd, workdir=finetune_dir)
            if not ok_bt:
                # Fix config and retry once for qlib_test
                current_text = config_path.read_text(encoding="utf-8")
                fixed_text = llm.fix_config_on_error(current_text, args.target, (out_bt or "") + "\n" + (err_bt or ""), detailed_context)
                config_path.write_text(fixed_text, encoding="utf-8")
                _ensure_run_id_in_file(config_path, run_id)
                llm_run_id = _extract_run_id_from_file(config_path) or run_id
                ok_bt, out_bt, err_bt = _launch(qlib_cmd, workdir=finetune_dir)
            if not ok_bt:
                print("[qlib_test] Failed after retry; continuing to next run.")
                snapshot_path = _snapshot_config(config_path, run_id)
                row = {
                    "ts": datetime.utcnow().isoformat(),
                    "run_id": run_id,
                    "target": args.target,
                    "init": args.init,
                    "error": (out_bt or "") + "\n" + (err_bt or ""),
                    "stage": "qlib_test",
                    "config_snapshot": str(snapshot_path),
                }
                _append_history(history_path, row)
                history.append(row)
                run_idx += 1
                print(f"[info] Completed run {run_id} with failure. Waiting {args.cooldown}s before next.")
                time.sleep(args.cooldown)
                continue

            kronos_cmd = [
                sys.executable,
                str(repo_root / "kronostest" / "kronostest.py"),
                "--run-id", run_id,
                "--signal-key", "all",
            ]
            ok_k, out_k, err_k = _launch(kronos_cmd, workdir=repo_root / "kronostest")
            if not ok_k:
                print("[kronostest] Failed; continuing to next run.")
                snapshot_path = _snapshot_config(config_path, run_id)
                row = {
                    "ts": datetime.utcnow().isoformat(),
                    "run_id": run_id,
                    "target": args.target,
                    "init": args.init,
                    "error": (out_k or "") + "\n" + (err_k or ""),
                    "stage": "kronostest",
                    "config_snapshot": str(snapshot_path),
                }
                _append_history(history_path, row)
                history.append(row)
                run_idx += 1
                print(f"[info] Completed run {run_id} with failure. Waiting {args.cooldown}s before next.")
                time.sleep(args.cooldown)
                continue
            backtest_summary = {"run_id": run_id, "backtest_dir": f"outputs/backtest_results/finetune_backtest_demo_{run_id}"}

        snapshot_path = _snapshot_config(config_path, run_id)
        row = {
            "ts": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "target": args.target,
            "init": args.init,
            "applied_updates": {},
            "applied_notes": {},
            "final_result": final_result,
            "summary_path": str(save_dir / "summary.json"),
            "config_snapshot": str(snapshot_path),
            "backtest": backtest_summary,
        }
        _append_history(history_path, row)
        history.append(row)

        # Next round relies on full-config generation; no partial updates needed
        run_idx += 1
        print(f"[info] Completed run {run_id}. Waiting {args.cooldown}s before next.")
        time.sleep(args.cooldown)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto tuning launcher for Kronos")
    parser.add_argument("--target", choices=["tokenizer", "predictor"], default="predictor")
    parser.add_argument("--init", choices=["pretrained", "scratch"], default="scratch")
    parser.add_argument("--nproc", type=int, default=1, help="Number of GPUs for torchrun")
    parser.add_argument("--history-path", type=str, default=str(Path(__file__).parent / "tuning_history.jsonl"))
    parser.add_argument("--llm-provider", type=str, choices=["openai", "vllm"], default="openai")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for qlib_test when target=predictor")
    parser.add_argument("--cooldown", type=int, default=5, help="Sleep seconds between runs")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        _run_pipeline(parse_args())
    except KeyboardInterrupt:
        print("[info] Stopped by user.")
