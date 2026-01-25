# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Whisper (audio -> text) via vLLM, with support for your own audio file.

Key idea: Whisper has a fixed decoder max length (~448), so for long audio
you must CHUNK the audio into shorter segments and transcribe each segment.
"""

import os
import time
from collections.abc import Sequence
from dataclasses import asdict
from typing import NamedTuple

import librosa
from vllm import LLM, EngineArgs, PromptType, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: Sequence[PromptType]


def run_whisper(audio_path: str, chunk_sec: float) -> ModelRequestData:
    # In WSL/spawn environments this is usually safer.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Whisper-large-v3-turbo in HF config has max_target_positions ~ 448.
    # Do NOT try to set max_model_len > 448 unless you really know what you're doing.
    engine_args = EngineArgs(
        model="openai/whisper-large-v3-turbo",
        max_model_len=448,
        max_num_seqs=16,
        limit_mm_per_prompt={"audio": 1},
        dtype="half",
    )

    # Whisper commonly uses 16kHz mono. Converting up-front is faster & more stable.
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    chunk_len = int(chunk_sec * sr)

    prompts: list[PromptType] = []
    for i in range(0, len(audio), chunk_len):
        chunk = audio[i : i + chunk_len]

        # Skip too-short tail (less than 1 second)
        if len(chunk) < sr:
            continue

        # For Whisper, these special tokens help set the mode (transcribe, no timestamps).
        prompts.append(
            {
                "prompt": "<|startoftranscript|><|transcribe|><|notimestamps|>",
                "multi_modal_data": {"audio": (chunk, sr)},
            }
        )

    return ModelRequestData(engine_args=engine_args, prompts=prompts)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Transcribe your audio file with Whisper via vLLM (offline)."
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        required=True,
        help="Path to your audio file (mp3/wav/ogg...). Example: /mnt/c/Users/Bulat/Downloads/test.mp3",
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=20.0,
        help="Chunk size in seconds (15-25 is usually safe). Default: 20",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when initializing vLLM.LLM",
    )
    return parser.parse_args()


def main(args):
    req_data = run_whisper(args.audio, args.chunk_sec)

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=256,            # per chunk
        skip_special_tokens=True,  # print clean text
    )

    start = time.time()
    outputs = llm.generate(req_data.prompts, sampling_params)
    duration = time.time() - start

    # Collect chunk texts and print as one transcript
    texts: list[str] = []
    for out in outputs:
        texts.append(out.outputs[0].text.strip())

    transcript = "\n".join(t for t in texts if t)
    print("\n===== TRANSCRIPT =====\n")
    print(transcript)

    # Perf stats
    if len(req_data.prompts) > 0:
        print("\n===== STATS =====")
        print("Chunks:", len(req_data.prompts))
        print("Duration:", duration)
        print("RPS:", len(req_data.prompts) / duration)

    # Avoid NCCL warning at exit (best-effort)
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
