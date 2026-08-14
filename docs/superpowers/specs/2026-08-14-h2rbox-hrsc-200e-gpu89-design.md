# H2RBox HRSC 200E GPU 8/9 Baseline Design

## Objective

Run a clean, reproducible pure-H2RBox baseline on HRSC2016 for 200 epochs on
physical GPUs 8 and 9. The run must isolate the original H2RBox consistency
loss and must not include any Orbdet harmonic-quality component.

## Experiment contract

| Item | Value |
|---|---|
| Detector | `H2RBoxDetector` with R50-FPN and `H2RBoxHead` |
| Consistency loss | Official `H2RBoxConsistencyLoss`, weight 0.4 |
| Supervision | HRSC qbox converted to enclosing hbox for training |
| Image size | 800 x 800 |
| Train split | `ImageSets/train.txt`, 436 images |
| Validation split | `ImageSets/val.txt`, 181 images |
| Held-out test | `ImageSets/test.txt`, 453 images |
| Physical GPUs | 8, 9 |
| Batch | 2 images/GPU, global batch 4 |
| Optimizer | AdamW, lr 5e-5, betas (0.9, 0.999), weight decay 0.05 |
| Warmup | Linear warmup for 500 optimizer iterations |
| Schedule | 200 epochs; LR decay at epochs 133 and 184, gamma 0.1 |
| Validation | Every 10 epochs on `val.txt` |
| Checkpoint selection | Highest validation `dota/mAP` |
| Test policy | Evaluate held-out test once after training using best-val checkpoint |
| Seed | 3407 |

The 200-epoch duration is user-selected. The optimizer and per-GPU batch follow
the closest official HRSC weak-supervision recipe discussed in the preceding
experiment review; schedule decay ratios preserve the 8/11 proportions of the
reference 12-epoch schedule.

## Files and isolation

Create dedicated 2-GPU formal and smoke configs and launchers. Use a new work
directory and log filename containing `gpu89`, `bs2`, `200e`, and the date so
no earlier bs8 or Orbdet-v0.1 artifact can be overwritten. Do not resume an old
checkpoint.

## Execution flow

1. Parse the merged MMEngine config and verify model type, loss type, split
   files, batch size, optimizer, scheduler, and work directory.
2. Run a bounded two-GPU smoke job on GPUs 8/9. Require finite loss, completion
   of all configured iterations, and a saved checkpoint.
3. Recheck GPU ownership. Start the formal 200E job only if GPUs 8/9 remain
   available and no matching baseline process already exists.
4. Launch with `CUDA_VISIBLE_DEVICES=8,9`, `NCCL_P2P_DISABLE=1`, and
   `NCCL_IB_DISABLE=1` in a dedicated tmux session.
5. Verify the launcher, both DDP ranks, GPU memory/utilization, finite early
   losses, and absence of OOM/NCCL/traceback errors.
6. Maintain a per-experiment Markdown record under
   `resultmd/exp_h2rbox_hrsc_baseline/`.

## Failure handling

- Do not kill or alter unrelated GPU processes.
- Abort this launch on OOM, non-finite loss, NCCL failure, incorrect merged
  config, or discovery that GPUs 8/9 are occupied before formal launch.
- Preserve all smoke and failed-launch evidence in separate directories.
- Do not silently change batch, learning rate, split, seed, or schedule after
  launch; any change is a new experiment ID.

## Success criteria

The launch is considered stable only after the smoke test passes and the formal
job shows both ranks training with finite losses and sustained GPU utilization.
Scientific success is not predefined by a target AP: the baseline result is the
measured best-validation checkpoint followed by one held-out test evaluation.

