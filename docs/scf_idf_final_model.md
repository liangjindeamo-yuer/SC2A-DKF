# SCF-IDF: Structure-Color Factorized Dynamic Kernel Filtering

本文档记录当前最终版模型架构：

```text
idf_grad8_rgb3d_scf_spatialbias_weak_stage1_k10
```

该模型以 `idf_grad8_rgb3d` 为基础，不再引入 WT-HF 子带恢复或自由 WGIDF controller，而是在原始 IDF 的图像域 iterative dynamic filtering 框架内，加入一个受约束的结构感知空间调制项。

核心思想是：

```text
保留 RGB-aware 3x3x3 dynamic kernel 的颜色建模能力；
额外学习一个 structure-aware 3x3 spatial modulation；
二者逐点相乘并重新归一化，得到最终动态滤波核。
```

---

## 1. Motivation

原始 IDF 的优势来自：

```text
1. image-domain iterative dynamic filtering
2. pixel-wise dynamic kernel
3. sum-to-one conservative filtering
4. Gaussian sigma=15 训练下良好的 OOD 泛化
```

前期 WT-HF / WGIDF 实验表明：

```text
1. 直接恢复小波高频子带容易破坏 IDF 的 OOD bias；
2. 自由 wavelet controller 会学成 Gaussian15 下的固定偏置；
3. 比较稳定的方向是保留 IDF 主干，只做小幅、可解释、可退化的结构调制。
```

因此最终模型选择：

```text
IDF image-domain backbone
+ Grad8 LCM
+ RGB-aware 3x3x3 kernel
+ weak structure-aware spatial modulation
```

---

## 2. Relation to Previous Models

### 2.1 Original IDF

原始 IDF 每次迭代预测一个动态空间滤波核：

```text
x_t -> dynamic kernel K_t -> x_{t+1}
```

其滤波核满足 sum-to-one 约束，使输出更接近保守滤波而不是自由回归。

### 2.2 idf_grad8_rgb3d

`idf_grad8_rgb3d` 在原始 IDF 上做了两个结构改动：

```text
1. LCM 使用当前迭代图像 x_t 的 8-direction gradient context；
2. 动态核从 channel-independent 3x3 扩展为 RGB-aware 3x3x3 kernel。
```

即对于每个像素 `p`：

```text
C_t(o, i, q, p)
```

其中：

```text
o: output RGB channel
i: input RGB channel
q: 3x3 spatial offset
p: pixel location
```

`C_t` 仍然经过 softmax / normalization，满足：

```text
sum_{i,q} C_t(o, i, q, p) = 1
```

### 2.3 Final SCF-IDF

最终版不改变 `idf_grad8_rgb3d` 的 RGB3D kernel 表达能力，而是在其上增加一个空间结构调制：

```text
S_t(o, q, p)
```

最终核为：

```text
K_t(o, i, q, p) = Normalize_{i,q} [ C_t(o, i, q, p) * S_t(o, q, p) ]
```

也就是说：

```text
RGB3D kernel C:
    负责颜色通道混合 + 空间邻域基础滤波

Spatial modulation S:
    负责基于局部结构，对 3x3 空间位置做弱调制

Final kernel K:
    C 和 S 相乘后重新归一化，仍保持 sum-to-one
```

---

## 3. Network Architecture

整体结构如下：

```text
Input noisy image x_0
    |
    v
for t = 0 ... T-1:
    x_t
      |
      +--> Feature Encoder / FEM
      |
      +--> Grad8 LCM from current x_t
      |
      +--> RGB3D Kernel Head
      |       -> C_t(o, i, q, p)
      |
      +--> Spatial Modulation Head
      |       -> Delta_t(o, q, p)
      |
      +--> Explicit Grad8 Spatial Bias
      |       -> B_t(q, p)
      |
      +--> S_t = softmax_q(Delta_t + B_t)
      |
      +--> K_t = Normalize(C_t * S_t)
      |
      +--> Dynamic filtering
              -> x_{t+1}

Output x_T
```

最终实验使用：

```text
T = 10 iterations
kernel size = 3x3
image channels = RGB
RGB-aware kernel size = 3x3x3
```

---

## 4. Grad8 LCM

LCM 使用当前迭代图像 `x_t` 计算 8 个方向的局部梯度：

```text
D = {
    up, down, left, right,
    left-up, right-up, left-down, right-down
}
```

方向梯度：

```text
G_d(p) = |x_t(p+d) - x_t(p)|
```

这些梯度作为局部结构上下文输入到 IDF 主干中，用于指导动态核预测。

注意：

```text
Grad8 LCM 从当前迭代图像 x_t 计算；
不是从 raw noisy image 固定计算。
```

这样可以让每一步滤波都根据当前恢复状态自适应更新。

---

## 5. RGB-Aware Dynamic Kernel

基础 RGB3D kernel 记为：

```text
C_t(o, i, q, p)
```

它表示在像素 `p` 处，输出通道 `o` 从输入通道 `i` 和空间偏移 `q` 取值的权重。

动态滤波形式为：

```text
y_o(p) = sum_i sum_q C_t(o, i, q, p) * x_i(p+q)
```

为了避免 color shift，RGB3D kernel 保留 diagonal prior / off-diagonal mixing gate：

```text
diagonal RGB path:
    主导同通道滤波

off-diagonal RGB path:
    只允许受控的小幅跨通道混合
```

这保留了颜色稳定性，同时允许模型在需要时利用 RGB 通道相关性。

---

## 6. Structure-Aware Spatial Modulation

最终版新增空间调制项：

```text
S_t(o, q, p)
```

它不是替代 RGB3D kernel，而是乘到原来的 RGB3D kernel 上。

### 6.1 Learned Spatial Delta

Spatial modulation head 输出：

```text
raw_delta_t(o, q, p)
```

经过幅度限制：

```text
Delta_t(o, q, p) = delta_max * tanh(raw_delta_t(o, q, p))
```

最终实验使用：

```text
delta_max = 0.2
```

因此空间调制只能小幅偏离初始均匀分布。

### 6.2 Explicit Spatial Gradient Bias

为了体现空间结构约束，引入显式梯度 bias：

```text
B_t(q, p) = - beta * normalize(|x_t(p+q) - x_t(p)|)
```

中心位置：

```text
B_t(center, p) = 0
```

含义：

```text
如果邻居和中心差异大，说明可能跨边缘；
该方向的 spatial weight 应该被轻微压低。
```

最终实验使用弱约束：

```text
beta = 0.1
```

### 6.3 Spatial Softmax

空间调制权重：

```text
S_t(o, q, p) = softmax_q(Delta_t(o, q, p) + B_t(q, p))
```

满足：

```text
sum_q S_t(o, q, p) = 1
```

---

## 7. Final Dynamic Kernel

最终动态核由 RGB3D kernel 和 spatial modulation 相乘得到：

```text
K_t(o, i, q, p)
  = C_t(o, i, q, p) * S_t(o, q, p)
```

然后在 `i,q` 维度重新归一化：

```text
K_t(o, i, q, p)
  = K_t(o, i, q, p)
    / sum_{i',q'} K_t(o, i', q', p)
```

因此最终仍满足 IDF 的保守滤波约束：

```text
sum_i sum_q K_t(o, i, q, p) = 1
```

最终更新：

```text
x_{t+1,o}(p)
  = sum_i sum_q K_t(o, i, q, p) * x_{t,i}(p+q)
```

---

## 8. Training Strategy

最终模型采用一阶段 conservative training。

### 8.1 Initialization

从当前最佳模型加载：

```text
runs/idf_grad8_rgb3d/CSVLogger/idf_grad8_rgb3d/checkpoints/last.ckpt
```

加载内容：

```text
1. IDF backbone
2. Grad8 LCM
3. RGB3D kernel head
```

新增部分：

```text
Spatial modulation head
```

### 8.2 Frozen Base

训练时冻结原有 `idf_grad8_rgb3d` 主干：

```text
requires_grad(base IDF + RGB3D kernel) = False
requires_grad(spatial_mod_head) = True
```

这样可以保证模型初始行为接近 `idf_grad8_rgb3d`，只学习小幅空间结构修正。

### 8.3 Data and Loss

训练协议与 IDF 主实验一致：

```text
Train dataset: CBSD432
Train noise: Gaussian sigma=15
Val dataset: CBSD68
Val noise: Gaussian sigma=15
OOD test: six noises
```

主 loss：

```text
L_rec = L1(x_out, x_clean)
```

空间调制正则：

```text
L_spatial_kl = KL(S || Uniform)
L_spatial_tv = TV(S)
```

最终：

```text
L = L_rec
  + 0.02 * L_spatial_kl
  + 0.001 * L_spatial_tv
```

没有使用：

```text
wavelet loss
amp loss
threshold TV
direction TV
mixed-noise training
```

### 8.4 Final Hyperparameters

```text
num_iter = 10
stage1_steps = 10000
spatial_grad_bias_beta = 0.1
spatial_delta_max = 0.2
spatial_mod_kl_weight = 0.02
spatial_mod_tv_weight = 0.001
```

---

## 9. Implementation Files

Main implementation:

```text
idf/archs/idf_structured_arch.py
```

Key modules:

```text
RGB3DSpatialModKernelHead
compute_spatial_gradient_bias
StructuredDIDBlock
IDFStructuredNet
```

Training script:

```text
scripts/train_idf_grad8_rgb3d_scf_spatialbias_weak_stage1.py
```

Final run:

```text
runs/idf_grad8_rgb3d_scf_spatialbias_weak_stage1_k10
```

Final OOD results:

```text
runs/idf_grad8_rgb3d_scf_spatialbias_weak_stage1_k10/ood_eval/step_last/six_noise_metrics.csv
```

---

## 10. Results

### 10.1 Six-Noise OOD Results

| Model | Gaussian50 | SpatialG55 | Poisson | S&P | Speckle | Mixture | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| IDF baseline | 25.7223 | 27.8840 | 27.3978 | 32.6993 | 29.1005 | 28.0499 | 28.4756 |
| idf_grad8_rgb3d | 25.8923 | 27.6560 | 27.7894 | 33.5688 | 29.5526 | 28.4698 | 28.8215 |
| Final SCF-IDF | 26.0053 | 27.7201 | 27.8195 | 33.0826 | 29.5407 | 28.4488 | 28.7695 |

### 10.2 SSIM Results

| Model | Gaussian50 | SpatialG55 | Poisson | S&P | Speckle | Mixture | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| IDF baseline | 0.7197 | 0.8014 | 0.7961 | 0.9024 | 0.8506 | 0.8156 | 0.8143 |
| idf_grad8_rgb3d | 0.7231 | 0.7902 | 0.8040 | 0.9187 | 0.8617 | 0.8252 | 0.8205 |
| Final SCF-IDF | 0.7262 | 0.7927 | 0.8034 | 0.9103 | 0.8603 | 0.8222 | 0.8192 |

### 10.3 Interpretation

Compared with original IDF:

```text
Average PSNR: +0.294 dB
Average SSIM: +0.0049
```

Compared with `idf_grad8_rgb3d`:

```text
Average PSNR: -0.052 dB
Average SSIM: -0.0013
```

The final SCF-IDF improves several dense or signal-dependent noises over `idf_grad8_rgb3d`:

```text
Gaussian50: +0.113 dB
SpatialG55: +0.064 dB
Poisson: +0.030 dB
```

But it is weaker on impulse-heavy Salt & Pepper:

```text
S&P: -0.486 dB vs idf_grad8_rgb3d
```

因此最终模型的主要收益是：

```text
更稳地提升 dense Gaussian / spatial variant / signal-dependent noise；
同时保持整体 OOD 结果明显高于原始 IDF。
```

---

## 11. Diagnostics

最终训练结束时的关键诊断量大致为：

```text
spatial_bias_mean ~= -0.10
spatial_mod_center_weight ~= 0.1155
spatial_mod_entropy ~= 2.188
spatial_mod_kl ~= 0.00016
spatial_mod_tv ~= 1.34e-05
```

解释：

```text
1. spatial bias 是弱负偏置，没有形成强中心抑制；
2. spatial modulation entropy 接近均匀 3x3 分布的 entropy；
3. KL / TV 很小，说明空间调制保持保守；
4. 模型没有重写 IDF，而是在 RGB3D kernel 上做小幅结构修正。
```

---

## 12. Final Method Summary

最终方法可以概括为：

```text
SCF-IDF keeps the original image-domain iterative dynamic filtering of IDF.
It uses Grad8 LCM for structure context and RGB-aware 3x3x3 dynamic kernels
for color-sensitive filtering. On top of the RGB3D kernel, it introduces a
weak structure-aware spatial modulation derived from local gradients. The
RGB3D kernel and spatial modulation are multiplied and renormalized, preserving
the sum-to-one conservative filtering bias of IDF.
```

中文总结：

```text
最终版不是一个小波高频恢复网络，也不是自由 controller。
它是一个保守的 IDF 结构增强版：

在 IDF 的 Grad8 + RGB3D 动态核基础上，
额外加入弱空间结构调制 S，
让滤波核在跨边缘方向轻微降权，
但始终保持 sum-to-one 和接近原模型的 OOD 归纳偏置。
```

