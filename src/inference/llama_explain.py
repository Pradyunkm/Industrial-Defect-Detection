"""Offline ExecuTorch Llama explanations for YOLO predictions."""

from collections import Counter
from pathlib import Path
import re
import subprocess

ROOT = Path.home() / "executorch-deployment"

RUNNER = (
    ROOT / "executorch/cmake-out/examples/models/llama/llama_main"
)

MODEL = (
    ROOT / "models/llama3.2-1b/Llama-3.2-1B-Instruct.pte"
)

TOKENIZER = (
    ROOT / "models/llama3.2-1b/tokenizer.model"
)

UNSUPPORTED = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|inches?|microns?|%)\b"
    r"|\b(caused by|due to|must be rejected|defect-free)\b",
    re.IGNORECASE,
)


def explain_defects(detections):
    """Return an evidence-based summary and optional Llama advice."""

    counts = Counter(
        str(item.get("class", "unknown"))
        for item in detections
    )

    if counts:
        details = ", ".join(
            f"{count} possible {name} region(s)"
            for name, count in sorted(counts.items())
        )
        summary = f"YOLO identified {details}."
    else:
        summary = (
            "YOLO found no regions above the selected threshold. "
            "This does not prove the surface is defect-free."
        )

    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a steel surface inspection assistant. "
        "Give one short visual inspection recommendation. "
        "Do not invent measurements, causes, severity, or pass/fail decisions."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + summary
        + " State one visual check in 20 words or fewer."
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    command = [
        str(RUNNER),
        "--model_path", str(MODEL),
        "--tokenizer_path", str(TOKENIZER),
        "--prompt", prompt,
        "--seq_len", "160",
        "--max_new_tokens", "40",
        "--temperature", "0",
        "--cpu_threads", "2",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )

    output = result.stdout.split("PyTorchObserver", 1)[0].strip()

    if output.startswith(prompt):
        output = output[len(prompt):].strip()

    # Remove Llama's end-of-message token.
    output = output.split("<|eot_id|>", 1)[0].strip()

    if not output or UNSUPPORTED.search(output):
        output = (
            "Visually inspect the marked regions and verify "
            "the findings against your inspection criteria."
        )

    return {
        "summary": summary,
        "llama_explanation": output,
    }
