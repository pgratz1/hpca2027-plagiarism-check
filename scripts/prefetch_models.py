"""One-time download of the sentence-embedding model into the local Hugging
Face cache (~/.cache/huggingface).  This is the ONLY script in this repository
that uses the network.  After it has run once, every comparison works with
HF_HUB_OFFLINE=1, which compare_pdfs.py sets for itself.

    python scripts/prefetch_models.py                       # default model
    python scripts/prefetch_models.py --model sentence-transformers/all-mpnet-base-v2
    python scripts/prefetch_models.py --check               # offline load test only

Only model weights travel over the network; no PDF or extracted text is ever
sent anywhere.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pdfsim.semantic import DEFAULT_MODEL  # noqa: E402  (sets offline env defaults)

OFFLINE_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
IGNORE = ["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.msgpack", "*.ot", "tf_model*", "rust_model*",
          "flax_model*"]

CHECK_CODE = (
    "import sys; sys.path.insert(0, sys.argv[2]); from pdfsim.semantic import load_model; "
    "m = load_model(sys.argv[1]); "
    "print('offline load OK:', sys.argv[1], '| dim', getattr(m, 'get_embedding_dimension', getattr(m, 'get_sentence_embedding_dimension', lambda: '?'))(), "
    "'| device', m.device)"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--check", action="store_true", help="only verify the model loads offline; no network")
    args = ap.parse_args()

    if not args.check:
        for k in OFFLINE_KEYS:
            os.environ.pop(k, None)
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=args.model, ignore_patterns=IGNORE)
        print(f"downloaded {args.model} -> {path}")

    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    proc = subprocess.run([sys.executable, "-c", CHECK_CODE, args.model, str(ROOT / "src")], env=env)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
