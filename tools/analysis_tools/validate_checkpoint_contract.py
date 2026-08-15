#!/usr/bin/env python3
"""Validate immutable training metadata before downstream evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--expected-epoch', type=int, required=True)
    parser.add_argument('--expected-iter', type=int, required=True)
    parser.add_argument('--expected-state-tensors', type=int)
    parser.add_argument('--config-token', action='append', default=[])
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()
    path = args.checkpoint.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        return fail(f'checkpoint is missing or empty: {path}')

    try:
        checkpoint = torch.load(path, map_location='cpu')
    except Exception as error:  # pragma: no cover - exact torch errors vary.
        return fail(f'checkpoint cannot be loaded: {error}')
    if not isinstance(checkpoint, dict):
        return fail('checkpoint root must be a dict')

    meta = checkpoint.get('meta')
    state_dict = checkpoint.get('state_dict')
    if not isinstance(meta, dict):
        return fail('checkpoint meta must be a dict')
    if not isinstance(state_dict, dict) or not state_dict:
        return fail('checkpoint state_dict must be a non-empty dict')

    epoch = meta.get('epoch')
    iteration = meta.get('iter')
    if epoch != args.expected_epoch:
        return fail(
            f'epoch mismatch: expected {args.expected_epoch}, got {epoch}')
    if iteration != args.expected_iter:
        return fail(
            f'iter mismatch: expected {args.expected_iter}, got {iteration}')

    state_tensors = len(state_dict)
    if (args.expected_state_tensors is not None
            and state_tensors != args.expected_state_tensors):
        return fail('state tensor count mismatch: expected '
                    f'{args.expected_state_tensors}, got {state_tensors}')

    config = meta.get('cfg')
    if not isinstance(config, str):
        return fail('checkpoint meta.cfg must be a string')
    for token in args.config_token:
        if token not in config:
            return fail(f'config token missing: {token}')

    evidence = dict(
        checkpoint=str(path),
        bytes=path.stat().st_size,
        epoch=epoch,
        iter=iteration,
        state_tensors=state_tensors,
        config_tokens=args.config_token,
        sha256=sha256(path))
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
