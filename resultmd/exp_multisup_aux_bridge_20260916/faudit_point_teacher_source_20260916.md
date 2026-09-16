# Point→OBB v2 教师来源核查（解 Phase-2 阻塞）

日期：2026-09-16。本轮**只读**：核读官方仓库源码/配置、抓取上游 README 与权重发布页。未训练、未下载权重、未改码、未动运行中的实验。

核查对象：`derived/whollywood_official_20260916`（commit `4a77e64`）+ 上游 `VisionXLab/point2rbox-v2` + 复现仓 `wokaikaixinxin/ai4rs`。

---

## 0. 结论先说

"Point→OBB v2" 有**两种教师语义**，必须分开说，否则会配错：

| 路线 | 是否需要教师 | 教师是什么 |
| --- | --- | --- |
| **Point2RBox-v2 端到端**（CVPR'25 主结果） | **不需要** | 无教师。监督由三个几何先验损失 + 合成图案提供 |
| **Point2RBox-v2 两阶段** | 需要 | **自己第一阶段训出的 Point2RBox-v2 模型**，教师产物是**伪标签 json**，不是教师模型张量 |
| **Wholly-WOOD 原仓 Point-to-RBox**（我们的 HOST） | 需要 | `Point2RBoxHDR`（s1）在训练集上跑 query mode 写出的 **`pseudo_labels/` 目录** |

**关键区分**：官方 point→OBB 的"教师"产出物是**伪标签文件**（弱监督目标），而我们现有 HBB `teacher_cache_c.pth` 是**教师模型推理缓存**（KD 张量）。两者不是同一种东西，**不可互换**。若要做 KD，教师模型应为 s1 的 `epoch_12.pth`。

---

## 1. 原仓（我们的 HOST）Point-to-RBox 的教师链条

配置与 README 均在本仓，为本轮实测核读：

**s1 = 伪标签生成**
- `configs/whollywood/whollywood-le90_r50_fpn-1x_dota-s1.py`
- `type='Point2RBoxHDR'`，head `Point2RBoxHDRHead`，`num_classes=15`，`strides=[8]`
- 数据侧：`RBox2Point(dummy=48, partial=1)`（纯点）
- 关键开关：`use_query_mode=True`，`query_mode_output_path='data/split_ss_dota/trainval/pseudo_labels'`
- 配套：`val_dataloader` 被重写为指向 **trainval**（`ann_file='trainval/annfiles/'`，`test_mode=True`）
- 即：**在训练集上跑推理，把预测 RBox 写成伪标签**

**s2 = 消费伪标签**
- `configs/whollywood/whollywood-le90_r50_fpn-1x_dota-s2.py`
- `type='H2RBoxV2PDetector'`，head `H2RBoxV2PHead`
- `train_dataloader.dataset.ann_file='trainval/pseudo_labels/'` ← 教师产物在这里进入训练
- 官方 DOTA-v1.0 结果：62.63 AP50（官方表，含 MS+RR 版 65.10）

**权重发布情况**：`configs/whollywood/metafile.yml` 中**只有 `h2rbox_v2` 的 5 个 openmmlab 权重**，**没有任何 whollywood / Point 路径的 Weights 项**。`docs/*/model_zoo.md` 里的下载链接全部是上游 MMRotate v0.1.0 的模型。

→ **结论：官方没有发布 Point 路径的任何权重。教师必须自己训 s1 产出。**

---

## 2. 上游 v2 仓

`VisionXLab/whollywood` 的 README 第 24 行明确：

> **Note: Our new work [Point2RBox-v2](https://github.com/VisionXLab/point2rbox-v2) is released, which also contains our series of work including Wholly-WOOD. This repository will remain as a record of the original Wholly-WOOD implementation.**

v2 仓的 Model Zoo 覆盖 Wholly-WOOD / Point2RBox-v2 / Point2RBox / H2RBox-v2 / H2RBox，但：

> Some of our methods are incorporated by [ai4rs](https://github.com/wokaikaixinxin/ai4rs) ... **Reproduced results and checkpoints can be found there.**

**本轮核读 `configs/point2rbox_v2/` 目录与 README，未见任何 checkpoint 下载链接，也没有 metafile.yml。** 即官方 v2 仓同样不发权重。

### 2.1 E2E 路线：教师被什么替代

`configs/point2rbox_v2/point2rbox_v2-1x-dota.py`（逐字核读）：

```python
model = dict(
    type='Point2RBoxV2',
    ss_prob=[0.68, 0.07, 0.25],
    copy_paste_start_epoch=6,
    ...
    bbox_head=dict(
        type='Point2RBoxV2Head',
        edge_loss_start_epoch=6,
        joint_angle_start_epoch=1,
        voronoi_type='standard',
        voronoi_thres=dict(default=[0.994, 0.005], override=(([2, 11], [0.999, 0.6]), ([7, 8, 10, 14], [0.95, 0.005]))),
        loss_cls=dict(type='mmdet.FocalLoss', loss_weight=1.0),
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
        loss_overlap=dict(type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
        loss_voronoi=dict(type='VoronoiWatershedLoss', loss_weight=5.0),
        loss_bbox_edg=dict(type='EdgeLoss', loss_weight=0.3),
        loss_ss=dict(type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0)))
```
pipeline 里 `ConvertWeakSupervision(point_proportion=1., hbox_proportion=0)` → 纯点。

**替代教师的是五个损失 + 两项课程式技巧**，全部无需任何外部权重或伪标签：
- `GaussianOverlapLoss`(10.0)：把物体当 2D 高斯，最小化重叠 → 学**上界**
- `VoronoiWatershedLoss`(5.0)：Voronoi 剖分上做 watershed → 学**下界**
- `Point2RBoxV2ConsistencyLoss`(1.0)：原图与增强视图两组输出的尺寸/旋转一致性
- `GDLoss(gwd)`(5.0) + `EdgeLoss`(0.3，第 6 epoch 起)
- `copy_paste_start_epoch=6`、`ss_prob=[0.68, 0.07, 0.25]`

官方 DOTA-v1.0：62.61 AP50（HRSC 86.15 / FAIR1M 34.71）。

### 2.2 两阶段路线：教师 = 自己第一阶段

`configs/point2rbox_v2/point2rbox_v2-pseudo-generator-dota.py`（逐字核读）：

```python
_base_ = 'point2rbox_v2-1x-dota.py'
model = dict(bbox_head=dict(pseudo_generator=True))   # 推理期可用 gt 点
test_pipeline = [..., dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0), ...]
test_dataloader = _base_.train_dataloader              # 在训练集上跑推理
test_dataloader['_delete_'] = True
test_evaluator = dict(_delete_=True, type='DOTAMetric', metric='mAP',
                      format_only=True,
                      outfile_prefix='data/split_ss_dota/point2rbox_v2_pseudo_labels')
```

流程：`point2rbox_v2-1x-dota.py` 训 12E → `epoch_12.pth` → 用上面的 generator 配置在 trainval 上推理 → 写出 `data/split_ss_dota/point2rbox_v2_pseudo_labels.bbox.json` → `rotated-fcos-1x-dota-using-pseudo.py`（s2 FCOS）消费它。

**所以 v2 的教师就是"自己第一阶段的 Point2RBox-v2 模型"，教师产物是伪标签 json。** 与第 1 节原仓结构同源，只是 s1 从专用 `Point2RBoxHDR` 换成了 Point2RBox-v2 本体。

---

## 3. 可下载的教师/权重来源（唯一发布处）

官方两个仓都不发权重。**唯一发布处是 ai4rs 的 ModelScope**，且**全部标注为 unofficial**：

| 产物 | 结果 | 官方对照 | 链接 |
| --- | --- | --- | --- |
| Point2RBox-v2 DOTA-v1.0 端到端 `epoch_12.pth` | 49.14 AP50 / 18.05 AP75 | 51.00 AP50 | `https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2-1x-dota/epoch_12.pth` |
| **DOTA-v1.0 伪标签 json（教师产物）** | — | — | `https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2_dotav1.0_rotated-fcos-1x-dota-using-pseudo/point2rbox_v2_pseudo_labels.bbox.json` |
| s2 FCOS `epoch_12.pth` | 59.72 AP50 / 25.96 AP75 | 62.61 AP50 | `https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2_dotav1.0_rotated-fcos-1x-dota-using-pseudo/epoch_12.pth` |
| DIOR-R 端到端 `epoch_12.pth` | 34.31 AP50 | 34.70 AP50 | `https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/Point2Rbox_v2/point2rbox_v2-1x-dior/epoch_12.pth` |

注意：ai4rs 的配置路径是 `projects/Point2Rbox_v2/configs/...`，与官方 v2 仓的 `configs/point2rbox_v2/...` **不同**，脚本不能直接混引。

DOTA-v1.0 端到端逐类 AP 已在 ai4rs README 给出（如 plane 0.7895、tennis-court 0.8845、harbor 0.2877、helicopter 0.1991）。

---

## 4. 意外发现：`ConvertWeakSupervision` 直接关系到我们的 C1/D2

**原仓没有这个类**（本轮实测：`grep -rn ConvertWeakSupervision` 于原仓返回空；原仓只有 `mmrotate/datasets/transforms/transforms.py` 的 `ConvertBoxType` / `RBox2Point`）。它是 **v2 新增**的混合监督 API：

```python
@TRANSFORMS.register_module()
class ConvertWeakSupervision(BaseTransform):
    def __init__(self, point_proportion=0.3, hbox_proportion=0.3,
                 point_dummy=1, hbox_dummy=0, modify_labels=False): ...

    def transform(self, results):
        max_idx_p = int(round(N * point_proportion))
        gt[0:max_idx_p, 2] = gt[0:max_idx_p, 3] = point_dummy   # 宽高→dummy
        gt[0:max_idx_p, 4] = 0                                   # 角度→0
        max_idx_h = max_idx_p + int(round(N * hbox_proportion))
        gt[max_idx_p:max_idx_h] = gt[max_idx_p:max_idx_h].convert_to('hbox').convert_to('rbox')
        gt[max_idx_p:max_idx_h, 4] = hbox_dummy
        # 余下 [max_idx_h:] 保持 RBox
        if modify_labels:   # ws_types: point=2, hbox=1, rbox=0
            results['gt_bboxes_labels'] = torch.stack((labels, ws_types), -1)
```

三点必须记录：

1. **按索引顺序切分，不是随机、无种子。** 官方取前 `round(N*p)` 行。这与我们 C1 实现的 **per-class 1:1 by seed2026 SHA256 哈希排序** 是**不同方案**。官方位置式；我们哈希式。二者都可复现，但语义不同：官方与标注行序相关，哈希式刻意打散该相关性。
2. **对"D2 fixed-mixed 1:1"有真实数值陷阱**：`point_proportion=0.5, hbox_proportion=0.5` 时，两个 `round` 独立取整（Python 银行家舍入），N 为奇数会出现残留。例 N=5：`round(2.5)=2` → point 2 行；`max_idx_h=2+2=4` → hbox 2 行；**第 5 行仍是 RBox**。即实际是 **2:2:1**，并非"1:1 全覆盖"。**若我们要宣称固定 1:1，必须显式处理这个残留**，否则协议名与内容不符。
3. **`modify_labels=True` 提供每行 ws_type 通道（point=2 / hbox=1 / rbox=0）** —— 这正好是我们 `AuxBatch` 需要按行携带的弱监督类型。**直接对齐它比从外部推断更干净**，建议 D1 采用。

---

## 5. 对 Phase-2 阻塞的更新

原阻塞表述"没有合法 Point 教师"仍然成立（当时无法伪造是正确的），但现在有**三条合法路径**，都不需要换强教师：

- **路径 A｜自训 s1（协议最干净）**：按官方配置训 Point2RBox-v2 E2E 12E（ResNet50 / DOTA1.0 trainval / batch2 / AdamW 5e-5），再用 pseudo-generator 导出伪标签。产物是确定性文件 → 可直接算 sha256 → 天然满足 `validate_teacher_scope` 的 manifest 要求。成本 = 一次 12E 训练。
- **路径 B｜下载 ai4rs unofficial 伪标签 json（最省算力）**：但必须在报告里声明三条限制：(i) unofficial；(ii) 复现分 49.14/59.72 vs 官方 51.00/62.61，差 1.9–2.9pp；(iii) 产物是 **COCO 风格 `.bbox.json`**，我们 `AnnotationRecord` 吃的是 DOTA annfiles 语义，**需显式转换并验证**。
- **路径 C｜走 E2E，不要教师**：若实验目标是"点监督能否学到 OBB"，Point2RBox-v2 单阶段即可，教师概念只在做 KD 时才有意义。

**但要注意**：Phase-2 的 D1/D2 是**KD 旁路**试验（教师给学生提供几何监督）。上述三条路径产出的是**伪标签（弱监督目标）**，不是"教师输出张量"。**要跑 D1/D2 的 KD，教师模型必须是 s1 的 `epoch_12.pth`（路径 A 的中间产物）**，而 ai4rs 那个 `epoch_12.pth` 是 unofficial 且与官方差 1.9pp，用不用需要单独决定并声明。

## 6. 本轮未做 / 待决

- 未下载任何权重或伪标签（含未验证 ModelScope 链接可达性）
- 未改动 `geometry_aux` / `gda_support_profile` / C1 协议代码
- 未动运行中的 `gda_object_equal_20260916`（epoch 8/12）
- 未 clone `ai4rs`（其 `ConvertWeakSupervision` 迁移与 `projects/Point2Rbox_v2/configs/` 需另行核读；且 `/data1` 已用 97%，clone 前需评估空间）
- **待用户决定**：走 A / B / C 哪条；若走 B，是否接受 unofficial 权重与 COCO→DOTA 转换链
