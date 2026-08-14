# 群轨道、秩与对称性：Orbdet 未来工作检索档案

## 1. 检索目标与结论

本轮检索针对一个具体问题：在水平框（HBox）内部，能否仅依靠矩阵秩、
旋转矩阵和人造物体的对称性，找到真实物体子区域并恢复其旋转方向；进一步，
哪些高等数学、线性代数或现代数学工具尚未被旋转目标检测充分利用。

截至 2026-08-15 的结论是：**直接搜索“旋转后秩最小的像素子矩阵”不能作为
可靠目标检测原理**。它容易被空白背景、常量区域、低纹理区域和任意秩一纹理
击败，而且真实对称物体在噪声、遮挡、插值和光照下通常具有满数值秩。

更有潜力的对象不是单张像素矩阵的秩，而是同一实例在候选有限群作用下形成的
**群轨道矩阵**。若候选群属于该实例的稳定子群，轨道成员应落入近似固定空间，
轨道矩阵会接近秩一。由此可构造可微、GPU 友好的小 Gram 矩阵损失。

本文是面向未来工作的研究档案，不是系统综述。检索采用 AI 辅助的定向检索，
覆盖至上述日期；“没有发现直接应用”不等价于证明不存在相关工作。

## 2. 为什么原始“最小秩子矩阵”假设不成立

设裁剪图像为 $A\in\mathbb{R}^{m\times n}$，$J_n$ 为列反转矩阵。
若图像具有精确竖直镜像对称性，

$$
A=A J_n,
$$

则成对列重复，因而

$$
\operatorname{rank}(A)\leq \lceil n/2\rceil.
$$

这说明某些对称性会给秩施加上界，但反命题不成立：

- 旋转 $180^\circ$ 对称的单位矩阵可以是满秩；
- 任意非对称外积 $uv^\top$ 都是秩一；
- 空白或常量背景天然是秩零或秩一；
- 轻微噪声、遮挡和双线性旋转插值通常把精确低秩提升为满秩；
- 子矩阵大小不固定会产生“越小越低秩”的尺度偏置。

因此，裸秩既不是对称性的充分条件，也不是必要条件。它必须和群作用、支持域、
能量/方差约束以及非对称负对照共同使用。

## 3. 与想法最接近的已有工作

| 方向 | 已有结果 | 对 Orbdet 的约束 |
|---|---|---|
| 变换 + 低秩 | TILT 已联合优化几何变换、低秩纹理和稀疏误差，并讨论旋转/反射对称扩展 | 不能把“搜索变换后最低秩区域”本身作为新颖性 |
| 旋转对称群检测 | Lee 与 Liu 已从图像中估计倾斜旋转对称群的中心、仿射形变、群型、阶数和支持域 | “检测哪个旋转群及其中心/支持”也已有强先例 |
| HBox 弱监督 OBB | H2RBox-v2 使用旋转与翻转自监督学习方向 | Orbdet 必须说明相对视图一致性的新增数学对象 |
| 对称先验 OBB | ABBSPO 已把对称先验用于方向优化 | 单纯加入对称正则不够形成明显创新 |
| 旋转不变/对称网络 | RIS-PiDiNet、Fourier Angle Alignment、结构张量等直接利用旋转结构 | 傅里叶、像素差和二阶矩都属于拥挤路线 |
| Radon 对称检测 | 已有基于 Radon 变换的反射对称检测 | Radon 不能单独作为核心新颖性 |
| 拓扑对称 | ECT/PHT 已用于对称检测，且已有可微 ECT | 拓扑特征适合作为辅助证据，不能声称无人用于学习 |

主要文献：

- [TILT: Transform Invariant Low-rank Textures](https://doi.org/10.1007/s11263-012-0515-x)
- [Skewed Rotation Symmetry Group Detection](https://doi.org/10.1109/TPAMI.2009.173)
- [H2RBox-v2](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html)
- [ABBSPO](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_ABBSPO_Adaptive_Bounding_Box_Scaling_and_Symmetric_Prior_based_Orientation_CVPR_2025_paper.html)
- [RIS-PiDiNet](https://openaccess.thecvf.com/content/CVPR2026/html/Zhan_Rotation_Invariant_and_Symmetry_Aware_Pixel_Difference_Network_for_Remote_CVPR_2026_paper.html)
- [Fourier Angle Alignment](https://arxiv.org/abs/2602.23790)
- [Structure Tensor OOD](https://arxiv.org/abs/2411.10497)
- [Radon-domain reflection symmetry](https://doi.org/10.1016/j.patcog.2022.108667)
- [Symmetry detection with the Euler characteristic transform](https://arxiv.org/abs/2307.08281)
- [Differentiable Euler Characteristic Transform](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c1fdec0d7ea1affa15bd09dd0fd3af05-Abstract-Conference.html)

## 4. 推荐数学核心：群轨道行列式簇

### 4.1 实例稳定子群

给定实例特征 $x$、候选有限群 $H$ 以及作用 $\rho(h)$，实例的稳定子群为

$$
\operatorname{Stab}(x)=\{h:\rho(h)x=x\}.
$$

不应预先假设所有人造物体都具有同一种对称性，而应在非平凡候选
$C_2,D_1,D_2,C_4$ 等之间估计实例级证据。无结构或不对称实例必须允许拒绝，
但不能让恒等群 $C_1$ 成为零损失捷径。

### 4.2 轨道矩阵与小 Gram 矩阵

对软支持 $m$ 内的特征构造

$$
x_h=\operatorname{vec}(m\odot\rho(h)x),\qquad
O_H=[x_h]_{h\in H},\qquad G=O_H^\top O_H.
$$

若 $H\subseteq\operatorname{Stab}(x)$，所有轨道列相同，故
$\operatorname{rank}(O_H)=1$。实际计算只需对 $|H|\leq 8$ 的 Gram 矩阵做
特征值分解，不需要对高维像素/特征矩阵做大型 SVD。

### 4.3 外代数/行列式秩一损失

令 $G\succeq0$，二阶外幂能量可由迹恒等式计算：

$$
L_{\wedge^2}=
\frac{(\operatorname{tr}G)^2-\operatorname{tr}(G^2)}
{2(\operatorname{tr}G)^2+\varepsilon}.
$$

除零能量退化外，$L_{\wedge^2}=0$ 当且仅当
$\operatorname{rank}(O_H)\leq1$。它等价于所有二阶子式平方和的归一化表达，
但无需显式枚举行列式。

谱尾可以作为互补量：

$$
L_{\mathrm{tail}}=1-
\frac{\lambda_{\max}(G)}{\operatorname{tr}G+\varepsilon}.
$$

### 4.4 Reynolds 固定空间残差

仅有秩一允许不同群元素产生互为标量倍数的特征。严格不变性还应使用 Reynolds
投影

$$
P_H=\frac{1}{|H|}\sum_{h\in H}\rho(h)
$$

或等价的轨道共识残差

$$
L_{\mathrm{fix}}=
\frac{\sum_h\lVert x_h-\bar{x}\rVert_2^2}
{\sum_h\lVert x_h\rVert_2^2+\varepsilon}.
$$

因此建议把“近似秩一”和“轨道成员确实相等”分开记录和消融。

### 4.5 稳定性而不是伪置信度

令 $\sigma_i=\sqrt{\lambda_i(G)}$，使用

$$
q_{\mathrm{gap}}=
\frac{\sigma_1-\sigma_2}{\sigma_1+\varepsilon}
$$

衡量秩一结构的谱稳定性。它只能作为诊断或门控候选，不能未经校准就解释为
检测正确率概率。

### 4.6 必须防止的退化解

- 零特征、空掩码和全背景；
- 常量特征或极低方差区域；
- 通过缩小支持域降低所有残差；
- 将任意实例分配给恒等群；
- 仅用自己产生的对称残差给自己加权；
- 方形/近圆实例的不可辨识方向被强行赋予唯一角度。

实现至少需要能量下限、方差下限、支持面积约束、非对称负对照以及 detach 的
稳定性诊断。角度输出应位于商空间
$S^1/\operatorname{Stab}(x)$，而不是强制一个欧氏标量真值。

## 5. 可继续深挖但不作为第一版核心的理论

### 5.1 FRI、Toeplitz/Vandermonde 与连续角度

将角度证据建模为圆上的有限脉冲流

$$
\mu=\sum_{\ell=1}^{s}a_\ell\delta_{\theta_\ell},\qquad
c_k=\sum_{\ell=1}^{s}a_\ell e^{-ik\theta_\ell}.
$$

对应 Toeplitz 矩阵满足 Vandermonde 分解并具有秩 $s$，可用 annihilating
filter、Matrix Pencil 或 ESPRIT 从少量谐波恢复连续角度。它更适合作为群轨道
模块之后的无栅格角度求解器，而非第一版核心损失。

- [Sampling Signals with Finite Rate of Innovation](https://doi.org/10.1109/TSP.2002.1003065)
- [Vandermonde decomposition of Toeplitz matrices](https://doi.org/10.1109/TIT.2016.2553041)

### 5.2 $p$-atic/整数模张量场

边界法向的 $p$ 阶序参量

$$
\psi_p=\int_{\partial E}e^{ip\theta_n}\,ds,
\quad q_p=\frac{|\psi_p|}{\int_{\partial E}ds},
\quad \vartheta_p=\frac{\arg\psi_p}{p}\pmod{2\pi/p}
$$

天然表达多重方向和角度商空间。$p=2$ 与结构张量较接近，因此更适合当作支持域
或对称阶数的辅助证据。

- [Integer-multiplied tensor fields](https://www.nature.com/articles/s42005-024-01751-1)
- [Applications of integer-multiplied tensors](https://elifesciences.org/articles/105680)

### 5.3 其他辅助方向

- 随机矩阵理论：用噪声谱边缘校准“有效秩”和阈值，而非固定经验阈值。
  参考 [Gavish--Donoho optimal singular-value shrinkage](https://doi.org/10.1109/TIT.2017.2653801)。
- Moment-SOS：为低分辨率角度/支持联合搜索提供可认证下界，适合离线教师而非
  在线检测头。参考 [Lasserre hierarchy](https://epubs.siam.org/doi/10.1137/S1052623400366802)。
- 拓扑签名：ECT/PHT 可提供遮挡下的形状证据，但计算和已有工作重叠较多。
- 商空间姿态：对不可辨识对称实例输出等价类或多峰分布。
  参考 [SARR](https://link.springer.com/article/10.1007/s11263-026-02770-x)。

## 6. 相对 Orbdet-v0.2 的实际创新空间

Orbdet-v0.2 当前只继承 H2RBox-v2 的原始/旋转/翻转角度一致性，并记录 detach
后的质量诊断。推荐的新模块与它有三个实质区别：

1. 对象从成对角度残差变为**实例特征的完整有限群轨道**；
2. 约束从标量角度一致性变为**固定空间 + 外代数秩结构**；
3. 输出从单一方向质量变为**群型、稳定性和商空间方向证据**。

第一阶段只实现通用数学核心及解析测试，不把它默认接入正式训练目标。这样可以
先证明损失实现与数学性质正确，再设计 HBox ROI 支持域和因果实验，避免一次性
混入特征裁剪、群选择和训练权重三个不确定因素。

## 7. 可证伪实验路线

1. 合成解析测试：精确 $C_2/D_1/D_2/C_4$ 对称、非对称、缩放秩一、零特征、
   常量特征、噪声和遮挡。
2. 模块消融：$L_{\wedge^2}$、$L_{\mathrm{tail}}$、$L_{\mathrm{fix}}$ 与退化保护。
3. 支持域消融：整 HBox、中心椭圆、软前景掩码、GT HBox ROI。
4. 负对照：随机纹理、背景裁剪和类别内非对称实例。
5. 识别性评估：群型准确率、轴角误差、谱间隙校准，不先看检测 AP。
6. 接入 Orbdet 后才做固定种子 smoke 和多种子验证；任何正式训练必须另行授权。

