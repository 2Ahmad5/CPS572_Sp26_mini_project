"""
Supervised Fine-Tuning (SFT) script.
Now saves BOTH sampler weights and training state to allow for RL stages.
"""

import argparse
import json
import os
import random
from typing import cast

import numpy as np
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

def _clean_text(value):
    if value is None: return None
    if not isinstance(value, str): return None
    return value.strip() or None

def _normalize_messages(messages):
    if not isinstance(messages, list): return None
    convo = []
    for msg in messages:
        if not isinstance(msg, dict): continue
        role = msg.get("role")
        if role not in {"system", "user", "assistant"}: continue
        content = _clean_text(msg.get("content"))
        if not content: continue
        convo.append({"role": role, "content": content})
    return convo if (convo and any(m["role"] == "assistant" for m in convo)) else None

def _gsm8k_answer_text(answer):
    text = _clean_text(answer)
    if not text: return None
    if "####" not in text: return text
    parts = text.split("####")
    reasoning = "####".join(parts[:-1]).strip()
    final = parts[-1].strip()
    return f"{reasoning}\n\nANSWER: {final}" if reasoning else f"ANSWER: {final}"

def _take_streaming_rows(dataset_name, split, n, seed, config_name=None):
    kwargs = {"path": dataset_name, "split": split, "streaming": True}
    if config_name: kwargs["name"] = config_name
    ds = load_dataset(**kwargs).shuffle(seed=seed, buffer_size=max(1000, n * 4))
    rows = []
    for row in ds:
        rows.append(row)
        if len(rows) >= n: break
    return rows

def main():
    parser = argparse.ArgumentParser(description="SFT Training with State Saving")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--num_steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--checkpoint_name", type=str, default="sft_model")
    parser.add_argument("--num_gsm8k", type=int, default=1000, help="Number of math examples (from MetaMathQA)")
    parser.add_argument("--num_tulu", type=int, default=1000)
    parser.add_argument("--num_opencode", type=int, default=1000, help="Number of code examples")
    parser.add_argument(
        "--code_dataset",
        type=str,
        default="opencode",
        choices=["opencode", "mbpp", "magicoder_python"],
        help="Code data source: opencode (default), mbpp, or python-only Magicoder",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.base_model)
    renderer_name = model_info.get_recommended_renderer_name(args.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    print("Loading data...")
    conversations = []
    # Math (MetaMathQA)
    rows = _take_streaming_rows("meta-math/MetaMathQA", "train", args.num_gsm8k, args.seed)
    for r in rows:
        question = _clean_text(r.get("query"))
        answer = _clean_text(r.get("response"))
        if question and answer:
            conversations.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
    # Tulu
    rows = _take_streaming_rows("allenai/tulu-3-sft-personas-instruction-following", "train", args.num_tulu, args.seed+1)
    for r in rows:
        convo = _normalize_messages(r["messages"])
        if convo: conversations.append(convo)
    # Code
    if args.code_dataset == "opencode":
        rows = _take_streaming_rows("nvidia/OpenCodeInstruct", "train", args.num_opencode, args.seed + 2)
        for r in rows:
            prompt_text = _clean_text(r.get("input"))
            code_text = _clean_text(r.get("output"))
            if prompt_text and code_text:
                conversations.append([
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": code_text},
                ])
    elif args.code_dataset == "mbpp":
        rows = _take_streaming_rows("google-research-datasets/mbpp", "train", args.num_opencode, args.seed + 2, "full")
        for r in rows:
            prompt_text = _clean_text(r.get("text"))
            code_text = _clean_text(r.get("code"))
            if prompt_text and code_text:
                conversations.append([
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": code_text},
                ])
    else:  # magicoder_python
        ds = load_dataset(
            "ise-uiuc/Magicoder-OSS-Instruct-75K",
            split="train",
            streaming=True,
        ).shuffle(seed=args.seed + 2, buffer_size=max(1000, args.num_opencode * 4))
        taken = 0
        for r in ds:
            if r.get("lang") != "python":
                continue
            prompt_text = _clean_text(r.get("problem"))
            code_text = _clean_text(r.get("solution"))
            if not prompt_text or not code_text:
                continue
            conversations.append([
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": code_text},
            ])
            taken += 1
            if taken >= args.num_opencode:
                break

    random.Random(args.seed).shuffle(conversations)
    all_data = []
    for convo in conversations:
        datum = conversation_to_datum(cast(list[renderers.Message], convo), renderer, max_length=args.max_length)
        if datum: all_data.append(datum)
    
    print(f"Prepared {len(all_data)} examples.")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    adam_params = types.AdamParams(learning_rate=args.lr)

    print(f"Training for {args.num_steps} steps...")
    for step in range(args.num_steps):
        start = (step * args.batch_size) % len(all_data)
        batch = [all_data[i % len(all_data)] for i in range(start, start + args.batch_size)]
        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        tc.optim_step(adam_params).result()
        fwd_bwd_result = fwd_bwd_future.result()
        step_logprobs = fwd_bwd_result.loss_fn_outputs[0]["logprobs"].to_numpy()
        step_loss = float(step_logprobs.mean())
        print(f"Step {step+1}/{args.num_steps} | Loss: {step_loss:.4f}")

    print("\nSaving Sampler Weights...")
    ckpt_sampler = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    print(f"Sampler Path: {ckpt_sampler.path}")

    print("Saving Training State (FOR RL)...")
    ckpt_state = tc.save_state(name=args.checkpoint_name).result()
    print(f"STATE PATH: {ckpt_state.path}")

if __name__ == "__main__":
    main()
