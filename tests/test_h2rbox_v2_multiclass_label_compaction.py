import torch

from mmrotate.models.dense_heads.h2rbox_v2_head import H2RBoxV2Head


def test_multiclass_label_compaction_preserves_integer_labels():
    labels = torch.tensor([9, 9, 9, 11, 11, 11, 2, 2, 2],
                          dtype=torch.long)
    group_index = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2],
                               dtype=torch.long)

    compacted = H2RBoxV2Head._compact_integer_labels(
        labels, group_index, num_groups=3)

    assert compacted.dtype == torch.long
    assert compacted.tolist() == [9, 11, 2]
