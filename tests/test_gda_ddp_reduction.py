"""CPU/Gloo contract for GDA distributed weighted means."""
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _unequal_count_worker(rank: int, init_file: str, output_dir: str):
    from mmrotate.models.losses.orbdet_gda_probe_losses import \
        _ddp_weighted_mean, _distributed_stat_mean

    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    dist.init_process_group(
        backend='gloo', init_method=f'file://{init_file}', rank=rank,
        world_size=2)
    try:
        parameter = torch.tensor(2.0, requires_grad=True)
        if rank == 0:
            local_sum = parameter + 2.0 * parameter
            local_count = parameter.new_tensor(2.0)
        else:
            # A zero-object rank must still participate in the collective and
            # keep the parameter connected to autograd.
            local_sum = parameter * 0.0
            local_count = parameter.new_tensor(0.0)
        loss = _ddp_weighted_mean(local_sum, local_count)
        loss.backward()
        grad = parameter.grad.detach().clone()
        dist.all_reduce(grad)
        grad /= dist.get_world_size()  # the averaging DDP applies
        torch.save(grad, Path(output_dir) / f'grad_{rank}.pt')
        stat_sum = parameter.new_tensor(2.0 if rank == 0 else 0.0)
        stat_count = parameter.new_tensor(2.0 if rank == 0 else 0.0)
        stat = _distributed_stat_mean(stat_sum, stat_count)
        torch.save(stat, Path(output_dir) / f'stat_{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_ddp_weighted_mean_matches_global_item_mean_with_zero_object_rank(
        tmp_path):
    init_file = tmp_path / 'gloo_init'
    mp.spawn(
        _unequal_count_worker,
        args=(str(init_file), str(tmp_path)),
        nprocs=2,
        join=True)
    grads = [torch.load(tmp_path / f'grad_{rank}.pt') for rank in range(2)]
    # Global reference is mean([1*p, 2*p]), whose derivative is 1.5.
    assert all(torch.allclose(grad, torch.tensor(1.5)) for grad in grads)
    stats = [torch.load(tmp_path / f'stat_{rank}.pt') for rank in range(2)]
    assert all(torch.allclose(stat, torch.tensor(1.0)) for stat in stats)
