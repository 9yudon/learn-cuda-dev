#!/usr/bin/env python3
"""Compile and benchmark candidate block sizes, then persist the winner as JSON."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys


def run(command: list[str], cwd: pathlib.Path | None = None) -> str:
    print("+", " ".join(command))
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default=pathlib.Path(__file__).parents[1])
    parser.add_argument("--build-root", type=pathlib.Path, default=pathlib.Path("build/autotune"))
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--architectures", default="80;86;89;90")
    args = parser.parse_args()

    source = args.source.resolve()
    root = args.build_root.resolve()
    results: list[dict[str, object]] = []
    for threads in (128, 256, 512):
        build_dir = root / str(threads)
        run([
            "cmake", "-S", str(source), "-B", str(build_dir), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
            "-DCUDA_OPS_BUILD_TESTS=OFF", f"-DCUDA_OPS_CUDA_ARCHITECTURES={args.architectures}",
            f"-DCUDA_OPS_RMSNORM_THREADS={threads}",
        ])
        run(["cmake", "--build", str(build_dir), "--target", "rmsnorm_bench", "--parallel"])
        executable = build_dir / ("rmsnorm_bench.exe" if sys.platform == "win32" else "rmsnorm_bench")
        output = run([str(executable), "--rows", str(args.rows), "--cols", str(args.cols),
                      "--iterations", str(args.iterations)])
        match = re.search(r"\{.*\}", output)
        if not match:
            raise RuntimeError(f"benchmark produced no JSON: {output}")
        result = json.loads(match.group())
        result["threads"] = threads
        results.append(result)

    winner = min(results, key=lambda item: float(item["latency_us"]))
    output_path = root / "rmsnorm_tuning.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workload = vars(args).copy()
    workload["source"] = str(source)
    workload["build_root"] = str(root)
    output_path.write_text(json.dumps({"workload": workload, "candidates": results, "winner": winner}, indent=2), encoding="utf-8")
    print(f"winner: {winner['threads']} threads, {winner['latency_us']} us")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
