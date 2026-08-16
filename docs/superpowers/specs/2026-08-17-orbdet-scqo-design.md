# Orbdet-SCQO：实例稳定子条件商空间方向学习设计

## 状态与授权边界

本方向于 2026-08-17 在对 RiO-DETR、Orbdet-v0.2、GODC 实验和既有数学检索
记录进行联合分析后获准写成设计稿。

本文只确定研究问题、数学对象、模块边界、实验顺序和晋级门槛。它不授权修改
模型代码、启动 smoke、正式训练、自动队列、<code>tmux</code>、
<code>nohup</code> 或恢复任务。用户审阅本设计后，下一步才是编写独立实施计划。

## 一句话决策

保留 Orbdet-v0.2 的 HBox-only 主干和当前推理路径，增加一个分阶段验证的
**Stabilizer-Conditioned Quotient Orientation（SCQO）** 分支：

1. 用 \(p=2\) 和 \(p=4\) 的谐波载体表达无向轴与四重方向；
2. 用停止梯度的群轨道、固定空间和 \(p\)-atic 证据估计实例方向是否可辨识；
3. 在由实例证据决定的商空间中施加跨视图等变损失；
4. 先证明方向表示和稳定子证据有效，再考虑替换主角度或增加正交双轴采样。

SCQO-v1 不把 GODC 继续作为通用 FPN 不变性正则，不改变分类分数、框分数、NMS
或最终角度解码。因此，新数学对象的有效性可以和推理结构变化分开归因。

## 研究背景与已有证据

### 当前可靠基线

Orbdet-v0.2 继承 H2RBox-v2 的原图、旋转图、翻转图三视图一致性，训练时通过
<code>qbox -&gt; hbox -&gt; rbox</code> 丢弃真实 OBB 方向，验证时才保留 OBB
标注。当前配置已启用独立角度分支
<code>use_standalone_angle=True</code>，角度主要由自监督一致性学习。

clean-HRSC 三个 seed 的 best-val mAP 为
<code>0.9029/0.8992/0.8977</code>，均值 <code>0.8999</code>，
population standard deviation 为 <code>0.0022</code>。因此小于约 0.22 个
AP 百分点的单次变化处在已有随机波动量级内。

### GODC 已给出的负结果

现有 GODC 在 GT-HBox FPN ROI 上施加 C2 群轨道的行列式、固定空间和退化保护。
它完成了 smoke 和 103E 训练，证明数值、梯度和工程路径稳定；但同 seed 最好
结果为 <code>0.8980</code>，低于 v0.2 的 <code>0.9029</code>，差值
为 <code>-0.0049</code>。

这不能证明群轨道思想无效，但足以否定“继续把固定空间损失直接叠加到通用 FPN
特征上”作为默认路线。SCQO 将现有 GODC 核心改作停止梯度的稳定子证据模块，
而不是让它直接优化或压平承载方向内容的通用特征。

### RiO-DETR 带来的约束

RiO-DETR 表明以下设计可以在完全监督 OBB DETR 中独立产生收益：

- 角度不进入几何位置查询，而由图像内容产生；
- 特征采样同时覆盖预测方向及其正交方向；
- 角度更新和损失遵守周期拓扑；
- 同一训练样本内的角度多样性可以加速角度学习。

Orbdet-v0.2 已部分具备“独立角分支”和最短周期残差，所以这两项只能作为
SCQO 的理论依据，不能作为新的贡献。可继续发展的交叉点是：把 RiO-DETR 的
固定 \(\pi\) 周期扩展成实例条件商空间，并用完整正交标架而非裸标量角度表达方向。

## 研究问题与可证伪假设

### 核心问题

在没有训练 OBB 方向标注的条件下，能否从同一实例的已知群作用视图中同时学习：

1. 一个稳定的无向主轴；
2. 当前实例的方向可辨识程度；
3. 当前实例应使用的角度等价类；
4. 对 AP75 和高精度定位真正有帮助、而非仅降低自监督残差的方向表示。

### 假设

- **H1：表示假设。** \(p=2\) 谐波载体可在 HBox-only 三视图监督下稳定学习，
  并且在相同特征和监督下，相对当前 PSC 分支提供更低的 GT-HBox ROI 角误差或
  更好的跨视图误差校准。
- **H2：稳定子假设。** 群轨道残差、谱结构和 \(p\)-atic 幅值的联合证据可以
  区分普通轴向实例、四重歧义实例和方向不可辨识实例；其信息量应显著超过简单
  的纹理能量或方差。
- **H3：商空间假设。** 按实例证据在 \(p=2\)、\(p=4\) 和 reject 之间选择
  一致性空间，可以降低角误差或提高 AP75，同时不破坏 AP50。
- **H4：双轴假设。** 只有当 SCQO 方向载体已经校准后，用其构造长轴/短轴双向
  采样才会带来额外收益；若方向载体未过门槛，正交采样不应实施。

任一假设均允许被单独否定。H1 成立而 H2 不成立时，项目保留谐波方向载体并
停止稳定子门控；H2 成立而 H3 不成立时，稳定子证据仅保留为诊断；H1 不成立时，
SCQO 主线停止，不进入双轴采样。

## 几何定义

### OBB 的基础商空间

长边 OBB 的无向轴天然满足

$$
\theta \sim \theta+\pi,
$$

因此普通 OBB 方向位于 \(S^1/C_2\)，而不是欧氏直线。Orbdet 现有 snap
residual 已经实现这一固定周期，但没有表达实例之间不同的可辨识性。

### 实例视觉稳定子

对实例特征 \(x\) 和图像作用 \(\rho(g)\)，视觉稳定子定义为

$$
\operatorname{Stab}(x)=\{g:\rho(g)x\approx x\}.
$$

第一版区分三个与 OBB 方向直接相关的状态：

| 状态 | 有效方向空间 | 训练载体 | 含义 |
|---|---|---|---|
| <code>regular</code> | \(S^1/C_2\) | \(p=2\) | 存在可辨识无向轴 |
| <code>quarter_turn</code> | \(S^1/C_4\) | \(p=4\) | 旋转 \(90^\circ\) 后近似等价 |
| <code>reject</code> | 不输出唯一方向证据 | 无 | 低能量、低方差、近连续对称或证据冲突 |

\(D_1/D_2\) 反射只在合成样本的已知规范坐标中做解析测试。真实旋转 ROI 上的
反射轴依赖尚未验证的实例方向，因此 SCQO-v1 不把 \(D_1/D_2\) 用作在线证据；
未来只有在 \(q_2\) 通过校准门槛后，才能另行设计方向校正后的反射证据。这样
避免把图像坐标反射误当成物体坐标反射，也避免把“反射相似”错误等同于“多出
一个旋转等价角”。

### 谐波方向载体

对 \(p\in\{2,4\}\)，模型输出二维向量

$$
q_p=(a_p,b_p),\qquad
u_p=\frac{q_p}{\lVert q_p\rVert_2+\varepsilon}
\approx(\cos p\theta,\sin p\theta).
$$

旋转图像 \(\phi\) 后，载体应满足

$$
u_p(T_\phi x)\approx R(p\phi)u_p(x).
$$

对 \(p=2\)，还可构造对称无迹张量

$$
Q_2=
\begin{bmatrix}
a_2 & b_2\\
b_2 & -a_2
\end{bmatrix}.
$$

其两个特征向量给出互相正交的长轴和短轴。这是未来双轴采样与 RiO-DETR
正交注意力之间的数学接口。

## 选定架构

~~~mermaid
flowchart LR
    V[原图/旋转图/翻转图] --> F[共享 Backbone + FPN]
    F --> B[原 Orbdet-v0.2 检测头]
    B --> P[原预测与损失保持不变]
    F --> H[谐波方向分支 q2/q4]
    H --> A[按 bid 聚合实例载体]
    V --> T[已知视图变换]
    T --> L[O(2) 等变损失]
    A --> L
    F -->|detach| R[14x14 方形 HBox ROI]
    R --> G[群轨道 + p-atic 证据]
    G --> W[detach 的 regular/C4/reject 权重]
    W --> L
~~~

### 1. 分层群作用视图采样器

保留现有三个视图及总体 3 倍 batch，不增加第四个图像视图：

- 原始中心裁剪；
- 每张图独立旋转；
- 垂直翻转。

当前实现为整个 batch 采样一个旋转标量。SCQO 候选将旋转范围仍限制在
\([\pi/4,3\pi/4]\)，但按 batch 内图像数分层采样并随机置换，各图不再共享同一
角度。采样不刻意落在 \(0,\pi/2,\pi\) 等离散角上，因为 \(\pi/2\) 对 \(p=4\)
载体是恒等相位，不能提供充分的四阶等变信号。

每个视图显式保存二维 \(O(2)\) 变换矩阵，而不只保存一个角度标量。这样旋转和
反射可以通过同一表示作用在 \(u_p\) 上。随机数继续服从 MMEngine/PyTorch seed，
保证 DDP 和恢复运行可复现。变换参数只进入损失，不作为模型输入，避免模型直接
读取监督答案。

### 2. 谐波方向分支

在现有 angle tower 的特征上增加每个 FPN 点四个通道：

$$
[q_{2x},q_{2y},q_{4x},q_{4y}].
$$

训练时沿用当前 <code>bid</code> 对应关系：先对同一视图、同一实例的所有正样本点
平均原始谐波向量，再归一化；随后把原图、旋转图、翻转图整理为
<code>[N,3,2]</code>。先平均后归一化可以避免单点方向噪声被单位化后等权放大。

SCQO-v1 中：

- 原 PSC 角度分支仍负责检测框角度和推理；
- 谐波分支只增加独立、可观察的辅助目标；
- 不使用 PSC 预测或 OBB 标注教师监督 \(q_2/q_4\)；
- 谐波输出不参与分类分数、bbox score、NMS 或后处理。

只有通过本文全部晋级门槛后，才为“由 \(q_2\) 解码角度并替换 PSC”另写设计。

### 3. 停止梯度的稳定子证据

每个训练实例从原始视图 FPN 提取以 HBox 中心为中心的方形 ROI：边长取 HBox
长边并裁剪到图像有效范围，RoIAlign 输出固定 <code>14x14</code>。采用固定二维
Hann 窗口作为软支持，避免可学习掩码通过缩小面积获得虚假低残差。

证据路径对输入 FPN 特征执行 <code>detach</code>，并做逐实例能量归一化。它复用
现有 GODC 小 Gram 核心，记录

$$
L_{\wedge^2,H},\quad L_{\mathrm{tail},H},\quad
L_{\mathrm{fix},H},\quad q_{\mathrm{gap},H},
$$

以及能量、方差、有效支持比例。第一版在线门控只使用 \(C_4\) 旋转轨道；
\(C_2\) 作为不改变门控的日志对照，\(D_1/D_2\) 仅用于合成规范坐标测试。
这些证据均不向检测网络反向传播。

另从停止梯度 ROI 的空间梯度构造归一化 \(p\)-atic 幅值：

$$
A_p=
\frac{\left|\sum_j w_j e^{ip\vartheta_j}\right|}
{\sum_j w_j+\varepsilon},
\qquad p\in\{2,4\},
$$

其中 \(\vartheta_j\) 来自各通道 Sobel 梯度合成的局部方向，\(w_j\) 为跨通道
梯度平方和的平方根乘固定 Hann 支持。结构张量或 \(p\)-atic 本身不作为新颖性
声明，只作为区别“有主轴”“四重结构”和“无方向证据”的独立辅助量。

每个真实 ROI 还生成一个保持通道直方图和能量、但使用固定 seed 做块级空间置换
的负对照。若原 ROI 的 C4 证据不优于该负对照，不能分配高
<code>quarter_turn</code> 权重。

证据映射遵守以下固定过程：

1. 用合成 \(C_2/C_4\)、非对称、圆形、零特征、常量、噪声和遮挡样本建立带标签
   的证据开发集；
2. 用冻结的 v0.2 checkpoint 对真实 train HBox ROI 做一次无标签前向，拟合每项
   证据的 median/MAD，此后连同 checkpoint 一起冻结；
3. 将稳健标准化后的 GODC、\(A_2/A_4\)、guard 和负对照 margin 输入三分类
   multinomial logistic calibrator；
4. calibrator 的 L2 系数只在合成开发/验证划分上选择，拟合完成后冻结；真实 OBB
   角度和类别级 rotation-agnostic 标记均不参与拟合。

最终输出三个和为一的连续权重

$$
(\pi_2,\pi_4,\pi_{\mathrm{reject}}),
$$

并在进入谐波损失前再次 <code>detach</code>。任何空支持、能量/方差低于解析
下限或非有限值的实例都被硬覆盖为 \((0,0,1)\)。若原 ROI 的 C4 证据不优于
负对照，只把 \(\pi_4\) 置零并在 \(\pi_2/\pi_{\mathrm{reject}}\) 间重新归一化，
不能因此拒绝具有清晰普通轴的实例。其余实例保留 calibrator 的连续 softmax；
若三个类别的最大概率低于 <code>0.5</code>，硬覆盖为 reject，作为“证据冲突”
的唯一实现定义。
\(q_{\mathrm{gap}}\) 不单独解释为正确率，也不允许给产生它的 GODC 残差自身加权。
calibrator 的冻结参数和 median/MAD 必须写入候选配置或 checkpoint 元数据；
恢复运行不得重新拟合。

### 4. 商空间等变损失

旋转视图的 \(p\) 阶损失为

$$
L^{\mathrm{rot}}_p
=1-\left\langle
u_p^{\mathrm{rot}},R(p\phi)u_p^{\mathrm{ori}}
\right\rangle.
$$

按照当前 H2RBox-v2 角度约定，垂直翻转对应
\(\theta\mapsto-\theta\)，因此

$$
L^{\mathrm{flip}}_p
=1-\left\langle
u_p^{\mathrm{flip}},
\operatorname{diag}(1,-1)u_p^{\mathrm{ori}}
\right\rangle.
$$

每阶损失保持现有翻转权重：

$$
L_p=L^{\mathrm{rot}}_p+0.05L^{\mathrm{flip}}_p.
$$

分阶段目标定义为：

- **表示探针阶段：** \(L_{\mathrm{SCQO}}=L_2\)，只验证 H1；
- **稳定子条件阶段：**
  \(L_{\mathrm{SCQO}}=\pi_2L_2+\pi_4L_4\)，reject 实例不贡献该辅助损失。

所有损失先按有效实例平均，再乘单一权重 \(\lambda_{\mathrm{SCQO}}\)。该权重不通过
AP 搜索：在无参数更新的梯度审计 batch 上一次性选择，使 SCQO 对共享 angle
tower 的梯度范数处于原 <code>loss_symmetry_ss</code> 梯度范数的
<code>[5%,10%]</code>，并限制在 <code>[0.01,0.10]</code>；随后整个实验冻结
此值。若不存在满足区间的值，候选在 smoke 前停止。

### 5. 未来正交双轴采样接口

若谐波方向载体通过晋级门槛，\(Q_2\) 的两个特征向量可生成方向分别为
\(\theta\) 与 \(\theta+\pi/2\) 的采样网格。未来候选应：

- 两组共享投影权重，避免参数量成为混杂变量；
- 初始对采样角执行 <code>detach</code>，防止错误角度通过采样路径自我强化；
- 同时覆盖长轴和短轴，再以固定平均或轻量门控融合；
- 单独报告 FLOPs、延迟和显存，不能照搬 RiO-DETR 的“无额外计算”结论。

该模块不属于 SCQO-v1 的实现范围。

## 数据流与监督隔离

### 训练

- HRSC/DOTA 原始 QBox 只在数据加载后转换为 HBox，再进入模型；SCQO 不读取
  被丢弃的真实 OBB 角度。
- 原图、旋转图、翻转图之间的已知 \(O(2)\) 变换和现有 <code>bid</code> 是唯一
  方向监督。
- 稳定子证据只读取停止梯度的 FPN ROI，不读取类别级
  <code>rotation_agnostic_classes</code>。
- 类别级静态旋转无关标记只用于事后对照，不训练实例稳定子。

### 验证

验证 OBB 标注只用于评估，不反向传播、不生成训练伪标签。评价分为两层：

1. **GT-HBox ROI 方向探针：** 排除检测召回误差，直接评估 \(q_2/q_4\) 与
   稳定子证据；
2. **端到端检测：** 保留原后处理，报告 AP50、AP75 和 AP50:95。

方向探针中的谐波输出仍来自 dense angle tower；GT-HBox 只用于复用训练时的
点分配和 <code>bid</code> 聚合，不另建带 OBB 信息的 ROI 方向头。稳定子证据则
使用同一 GT-HBox 的停止梯度 FPN ROI。

HRSC 主要是船类，只足以验证 \(p=2\) 表示和非退化；实例稳定子与
<code>quarter_turn/reject</code> 的主要证据必须来自具有独立验证划分的 DOTA
类数据。当前 DOTA 12E checkpoint 没有独立 validation 指标，不能用 trainval
自评替代。相关候选必须使用 DOTA 官方 train 训练、官方 val 选择与评价，不复用
当前 trainval 合并训练合同作为因果基线。

### 测试集

HRSC test 和 DOTA 官方 test 在架构、损失权重、阈值和 checkpoint 选择全部冻结
之前保持未触碰。任何最终 test 评估需单独授权。

## 评价指标

### 方向表示

从载体解码只用于验证诊断：

$$
\hat\theta_2=
\frac{1}{2}\operatorname{atan2}(b_2,a_2)
\pmod{\pi},
\qquad
\hat\theta_4=
\frac{1}{4}\operatorname{atan2}(b_4,a_4)
\pmod{\pi/2}.
$$

对普通无向轴，使用

$$
e_2=
\left|
\operatorname{wrap}_{[-\pi/2,\pi/2)}
(\hat\theta-\theta_{\mathrm{gt}})
\right|.
$$

报告 mean、median、P90，以及 \(5^\circ/10^\circ/15^\circ\) 内比例。对四重
歧义诊断另报告

$$
e_4=
\min_{k\in\mathbb Z}
\left|
\hat\theta-\theta_{\mathrm{gt}}-k\pi/2
\right|.
$$

### 稳定子和置信度

- 用证据预测 \(e_2>15^\circ\) 的 AUROC、AUPRC；
- 三折交叉校准后的 ECE 和可靠性曲线；
- 与单独使用 ROI 能量、方差、长宽比的简单基线比较；
- 按纹理强弱、长宽比、框面积和类别分桶；
- 检查 reject 是否集中在低信息实例，而不是集中在某个尺度或 FPN 层。

### 检测

- AP50：保证基础召回和粗定位不退化；
- AP75、AP50:95：检验方向改善是否转化为高精度框；
- 每类 AP 与按长宽比分桶的 rotated IoU；
- 参数量、训练显存、训练吞吐和推理延迟。

## 分阶段实验矩阵

| 编号 | 唯一变化 | 目的 | 是否改变推理 |
|---|---|---|---|
| E0 | Orbdet-v0.2 原基线 | 固定参照 | 否 |
| E1 | 每图分层旋转采样 | 验证角度覆盖原则 | 否 |
| E2 | \(q_2\) 表示探针，原采样 | 验证 H1 | 否 |
| E3 | \(q_2\) + 分层采样 | 验证采样与载体交互 | 否 |
| E4 | \(q_2/q_4\) + detach 稳定子条件 | 验证 H2/H3 | 否 |
| E5 | 通过后再设计双轴采样 | 验证 H4 | 是，另立设计 |

E1 与 E2 分开，避免把采样变化和表示变化混成一次实验；E4 不同时更换 PSC
解码或检测后处理。

## 晋级与停止门槛

### A. 解析与模块门槛

1. 旋转/反射群作用在 \(p=2,4\) 上通过解析等式测试；
2. 常量输出不能在连续分层旋转上获得零损失；
3. C4、普通轴向、圆形/无方向、零特征、噪声和遮挡合成样本被正确区分；
4. 空 batch、无正样本、空支持、半精度和非有限输入均有明确有限行为；
5. 稳定子权重和所有诊断均无梯度，不能自我操纵门控；
6. 原 v0.2 prediction path 的逐元素回归测试保持一致；
7. 匹配尺度的背景 ROI、只含插值边界的 ROI 和随机纹理 ROI 不能获得与真实前景
   相同的载体幅值与角度校准。

### B. 既有 checkpoint 的只读证据门槛

复用现有 v0.2/GODC checkpoint，不训练新模型。组合稳定子证据预测
\(e_2>15^\circ\) 的 AUROC 必须：

- 至少达到 <code>0.65</code>；
- 比最佳单变量能量/方差基线高至少 <code>0.05</code>；
- 在低、中、高纹理三桶中方向一致，不能仅由最低纹理桶贡献。

未通过时，E4 取消；E2/E3 仍可独立进行。

### C. 单 seed 筛选门槛

所有正式训练均需另行授权。首个 seed 只用于筛选：

- AP50 相对同 seed E0 不低于 <code>-0.2</code> 个百分点；并且
- AP75 至少提高 <code>0.3</code> 个百分点，或 GT-HBox ROI 的 median
  \(e_2\) 至少降低 <code>10%</code> 且 P90 至少降低 <code>5%</code>。

只降低自监督损失而不改善方向误差或 AP75，判定为失败。

### D. 多 seed 晋级门槛

通过单 seed 后才运行与 v0.2 相同的 <code>3407/42/2026</code> 三 seed 合同。
候选需满足：

- mean AP50 提高至少 <code>0.4</code> 个百分点；或 mean AP75 提高至少
  <code>0.5</code> 个百分点且 mean AP50 非劣于 <code>-0.2</code>；
- worst-seed AP50 不低于对应基线 <code>0.3</code> 个百分点以上；
- 三个 seed 的方向误差改善方向一致。

### E. 稳定子结论门槛

HRSC 只能支持 H1。要声称 H2/H3，必须在独立 DOTA validation 上证明：

- 实例证据相对静态 <code>rotation_agnostic_classes</code> 能更好预测角度高误差
  或低 OBB IoU；
- <code>quarter_turn/reject</code> 不只是类别分类器，至少一个相关类别内存在
  可复现的实例级差异；
- 对普通细长类 AP75 不退化，对近方形/旋转无关类的错误角惩罚或框 IoU 改善。

若只能复现类别级静态掩码效果，不形成实例级证据，则不能作为核心创新声明。

## 退化解与处理

| 退化模式 | 处理 |
|---|---|
| \(q_p=0\) | 归一化前记录幅值；零幅值产生有限惩罚并不能通过解析测试 |
| 所有实例都预测同一方向 | 连续分层旋转使常量载体违反 \(R(p\phi)\) 等变关系 |
| 所有实例进入 reject | reject 由停止梯度解析证据决定，不由可学习 logits 自由选择 |
| 背景或低纹理呈低秩 | 能量、方差、支持下限和负对照共同拒绝 |
| 可学习掩码缩小支持 | 第一版只用固定 Hann 支持，不提供可学习面积 |
| C4 伪对称来自插值 | 同时检查原 ROI、负对照和多证据一致性，不只看一个残差 |
| \(q_{\mathrm{gap}}\) 自我强化 | 全路径 detach，且不允许它重加权自身 GODC 损失 |
| 模型从旋转插值或裁剪边缘猜变换 | 变换参数不进模型；加入背景 ROI、替代插值和边界消融，前景优势不过门即停止 |
| 角度错误导致采样错误 | SCQO-v1 不做方向条件采样；未来双轴阶段先 detach 角度 |
| 使用验证 OBB 调阈值 | 稳定子阈值先在合成开发集冻结，真实 OBB 仅作评价 |

## 模块边界与预期文件

下一份实施计划应优先复用现有模块。文件名可以在计划阶段根据 MMRotate 注册结构
做机械性调整，但职责不得合并：

| 职责 | 预期位置 |
|---|---|
| \(O(2)\) 的 \(p=2/4\) 表示与损失 | <code>mmrotate/models/losses/scqo_harmonic_equivariance_loss.py</code> |
| detach ROI、群轨道与 \(p\)-atic 证据 | <code>mmrotate/models/task_modules/scqo_stabilizer_evidence.py</code> |
| 每点谐波输出与 bid 聚合 | <code>mmrotate/models/dense_heads/scqo_h2rbox_v2_head.py</code> |
| 保持原 prediction 的薄 detector 子类 | <code>mmrotate/models/detectors/orbdet_scqo.py</code> |
| GT-HBox ROI 角度/校准评价 | <code>mmrotate/evaluation/metrics/scqo_diagnostic_metric.py</code> |
| 解析、退化和回归测试 | <code>tests/test_scqo_*.py</code> |
| 独立候选配置 | <code>configs/orbdet/orbdet_scqo_*.py</code> |

必须复用：

- 现有 <code>bid</code> 三视图对应关系；
- 现有 HBox-only 数据合同；
- <code>GroupOrbitDeterminantalClusterLoss</code> 的 Gram、固定空间和 guard；
- v0.2 当前 PSC/angle prediction path；
- 现有独立角度周期残差作为基线。

不得把 SCQO 逻辑塞入 GODC 核心或原
<code>H2RBoxV2ConsistencyLoss</code>，以免新旧实验无法隔离。

## 实施拆分

本设计覆盖完整研究链，但不应由一份实施计划一次完成。按因果边界拆成三份：

1. **计划 A：只读诊断与解析核心。** 实现方向指标、合成群作用测试、现有
   checkpoint 证据抽取和 B 门槛；不训练模型。
2. **计划 B：表示探针。** 只有计划 A 完成后，实施 E1/E2/E3 的代码、smoke 和
   单 seed 筛选；任何正式训练仍需用户在当轮授权。
3. **计划 C：实例稳定子与 DOTA。** 只有 H1 和 B 门槛均通过后，实施 E4 及
   DOTA official train-to-val 合同。

E5 正交双轴采样不属于上述三份计划，必须重新设计和批准。

## 备选方案与取舍

### 采用：并行谐波分支 + detach 稳定子证据

优点是 HBox-only 属性、当前推理结果和已有基线全部保持可比；失败时可以精确判断
是表示、证据还是门控问题。代价是第一阶段不会立即获得推理速度或 AP 的保证。

### 不采用：直接用 \(q_2\) 替换 PSC

表示变化、角度解码、框回归和 NMS 会同时变化，无法判断收益来源。它只在
SCQO-v1 过门槛后单独设计。

### 不采用：把 GODC 固定空间损失继续加大或扩到 C4/D2

已有同 seed 结果未显示精度收益。扩大群或权重只会增加通用特征被过度不变化、
支持域偏置和低纹理捷径风险。

### 不采用：迁移至 RiO-DETR/DETR 主干

这会把研究问题从 HBox-only CNN 方向学习改成完全监督 OBB DETR，同时引入匹配、
decoder 和基线差异；RiO-DETR 的论文核心代码也未完整开放。当前只迁移其几何
原则，不迁移架构。

### 延后：FRI/Toeplitz/Vandermonde 多峰角度求解

它适合在证明确有多峰、连续角度证据后作为无栅格求解器。第一版加入会同时引入
阶数选择、数值条件和复根配对问题，超出当前可证伪范围。

## 预期论文贡献边界

若全部门槛通过，可以尝试的贡献表述是：

> 在 HBox 弱监督二维旋转检测中，从实例特征估计方向可辨识性，并在实例条件的
> 旋转商空间中学习方向；同时显式分离常规轴向、四重歧义与拒绝状态。

不能单独声称为新颖性的内容包括：

- 正弦/余弦或 \(p=2\) 编码；
- 一般周期角损失；
- 一般旋转/翻转一致性；
- 一般对称正则、结构张量或低秩；
- 一般已知对称类别的商空间姿态表示；
- RiO-DETR 的正交注意力或 Dense O2O。

真正需要文献查重和实验共同支撑的是四者组合：

1. HBox-only OBB；
2. 未知的实例级稳定子/可辨识性证据；
3. \(p=2/4\) 条件商空间一致性；
4. 不变性证据与等变方向载体的显式隔离。

在完成面向 stabilizer-aware orientation、quotient pose、
symmetry-aware HBox OBB 和 p-atic detection 的系统查重前，不使用“首次”表述。

## 安全与复现

- 不改写 <code>data/DOTA-v1.0/</code> 或 HRSC 源数据；
- 每个候选使用独立 config、work directory 和结果记录；
- smoke 与正式训练严格分开，且正式训练必须由用户在当轮明确授权；
- 物理 GPU 8/9 运行时必须设置
  <code>CUDA_VISIBLE_DEVICES=8,9</code>、
  <code>NCCL_P2P_DISABLE=1</code>、
  <code>NCCL_IB_DISABLE=1</code>；
- 不终止或修改其他用户的 GPU 进程；
- <code>FORMAL_TRAINING_NOT_STARTED.md</code> 只按真实运行状态更新，不能因
  设计、测试或只读评价改变；
- 所有结论区分解析测试、单 seed 观察、多 seed 证据和独立数据集泛化。

## 参考与项目证据

- RiO-DETR：<https://arxiv.org/html/2603.09411v2>
- RiO-DETR 官方仓库：<https://github.com/RicePasteM/RiO-DETR>
- H2RBox-v2：<https://proceedings.neurips.cc/paper_files/paper/2023/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html>
- SARR：<https://link.springer.com/article/10.1007/s11263-026-02770-x>
- 数学检索档案：
  <code>docs/future_work/inspirations/2026-08-15-group-orbit-determinantal-cluster.md</code>
- GODC 数学核心设计：
  <code>docs/superpowers/specs/2026-08-15-group-orbit-determinantal-cluster-design.md</code>
- v0.2/GODC 实验结果：
  <code>resultmd/exp_orbdet_overnight_123_20260815/fres_orbdet_overnight_123_gpu89.md</code>
