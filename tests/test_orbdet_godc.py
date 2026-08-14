import torch
from mmengine.structures import InstanceData

from mmrotate.models.detectors.h2rbox_v2 import H2RBoxV2Detector
from mmrotate.models.detectors.orbdet_v0_2 import OrbdetV02Detector
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.utils import register_all_modules


def _adapter_cfg():
    return dict(
        type='HBoxFPNGroupOrbitLoss',
        group='c2',
        loss_weight=0.02,
        min_box_size=2.0,
        roi_extractor=dict(
            type='mmdet.SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=4,
            featmap_strides=[1]),
        orbit_loss=dict(
            type='GroupOrbitDeterminantalClusterLoss',
            energy_guard_weight=0.0,
            variance_guard_weight=0.0))


def _instances():
    instances = InstanceData()
    instances.bboxes = RotatedBoxes(
        torch.tensor([[10., 10., 12., 8., 0.]]))
    instances.labels = torch.zeros(1, dtype=torch.long)
    return instances


def test_h2rbox_v2_default_auxiliary_hook_is_exact_noop():
    losses = {'loss_bbox': torch.tensor(1.0)}

    returned = H2RBoxV2Detector._add_auxiliary_losses(
        None, losses, (), [])

    assert returned is losses
    assert list(returned) == ['loss_bbox']


def test_godc_detector_inherits_v02_and_preserves_prediction_path():
    register_all_modules()
    detector_cls = MODELS.get('OrbdetGODCDetector')

    assert detector_cls is not None
    assert issubclass(detector_cls, OrbdetV02Detector)
    assert 'predict' not in detector_cls.__dict__


def test_godc_auxiliary_hook_emits_one_loss_and_detached_diagnostics():
    register_all_modules()
    detector_cls = MODELS.get('OrbdetGODCDetector')
    detector = object.__new__(detector_cls)
    torch.nn.Module.__init__(detector)
    detector.godc_auxiliary = MODELS.build(_adapter_cfg())
    feature = torch.randn(1, 4, 20, 20, requires_grad=True)
    losses = {'loss_bbox': feature.sum() * 0.0}

    returned = detector._add_auxiliary_losses(
        losses, (feature, ), [_instances()])
    total = returned['loss_godc'] + returned['loss_bbox']
    total.backward()

    assert returned is losses
    assert returned['loss_godc'].shape == ()
    assert torch.isfinite(returned['loss_godc'])
    assert float(returned['loss_godc']) > 0.0
    diagnostic_names = [name for name in returned if name.startswith('godc_')]
    assert diagnostic_names
    assert all('loss' not in name for name in diagnostic_names)
    assert all(not returned[name].requires_grad for name in diagnostic_names)
    assert feature.grad is not None
    assert float(feature.grad.abs().sum()) > 0.0
