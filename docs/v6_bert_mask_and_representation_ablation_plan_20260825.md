# V6 BERT mask、表示目标与下游提升消融计划

日期：2026-08-25

## 1. 当前判断

当前最可信的统一基线仍是：V6、标准 FSQ `[9,7,5,5]`、BERT 分维度 CE、左右眼同步 mask、只 mask 双眼均有效且非 padding 的时间 patch、`span[1,5]` 长度均匀、mask ratio 0.60；下游使用 strict four-task、每任务 K16、shared trial-logit mean、冻结 embedding 并解冻顶部 8 层。

现有结果不能支持“BERT loss 越低或 code accuracy 越高，下游一定越好”。历史受控和半受控结果显示：

| BERT 方案 | MCI Val/Test AUROC | PD5 Val/Test macro-AUROC | 解释 |
|---|---:|---:|---|
| span `[2,6]`, ratio .25, joint CE | .8884 / .8805 | .8606 / .8719 | 预测任务较容易 |
| span `[2,6]`, ratio .50, joint CE | .8982 / .8834 | .8608 / .8684 | MCI 改善，PD5 无改善 |
| span `[1,6]`, symmetric, ratio .50 | .8672 / .8724 | .8687 / .8699 | PD5 较好，MCI 明显下降 |
| span `[1,6]`, symmetric, ratio .60 | .9146 / .8875 | .8586 / .8699 | MCI Val 高，PD5 下降 |
| span `[1,5]`, uniform, ratio .60, factorized CE | .9017 / .8810 | .8626 / .8810 | 当前最简洁的统一方案 |
| span `[1,8]`, uniform, ratio .60, factorized CE, 80K | .8880 / .8951 | .8664 / .8663 | 更长 span/更多步数无一致收益 |

不同实验还混有 batch、训练步数、joint/factorized head 和 loss reduction 差异，因此上表用于确定搜索方向，不能代替严格因果消融。最关键的空缺是：在同一 tokenizer、同一分维度 CE、同一 batch/LR/steps 下，严格比较 random 与 span，以及 ratio 0.40/0.50/0.60。

### 当前主要瓶颈

1. **小规模 subject validation 方差。** MCI validation 约 60 人，单 seed AUROC 很容易因少量 subject 排序变化而波动；最高单次 Val 不是稳定最优的证据。
2. **MCI 和 PD5 对上下文尺度偏好不同。** MCI 更可能依赖跨时间、跨任务的认知控制模式；PD5 更依赖局部运动形态。更强连续遮挡有利于前者，却可能抹掉后者所需的局部锚点。
3. **离散目标存在 tokenizer 上限。** Code CE 只要求恢复 tokenizer 的类别，不保证保留所有与疾病相关的连续细节；但 raw reconstruction 又可能把容量浪费在低层幅值，必须直接做对照。
4. **下游优化噪声仍大于许多结构改动的收益。** K16 已比 K4 更稳；继续增加一致性 loss、复杂残差融合或更大分类头，历史上主要改变置信度而非 subject 排序。

因此，不应继续无控制地增加 span、mask ratio 或模型复杂度。先做少量正交、可归因的实验；只有下游 validation 的多 seed 结果通过门槛，才进入完整训练。

## 2. 先提高下游任务，而不重训 BERT

这些实验成本最低，应先锁定，随后所有 BERT 候选使用同一协议。

### 2.1 固定项

- strict four-task；任一任务不足训练门槛则该 subject 不进入 strict 训练。
- 每任务 K16，无放回；验证和测试使用所有有效 trial。
- 每个 trial 独立得到 shared-head logit；先在任务内平均，再四任务等权平均。
- demographics 为原始 16D 向量，与 `LayerNorm(CLS)` 拼接。
- MCI 使用 subject positive weight、阈值 0.5；PD5 使用 inverse-frequency class weight、argmax。
- validation 选 best checkpoint；test 不参与选型。

### 2.2 只保留的小网格

| 轴 | MCI | PD5 | 原因 |
|---|---|---|---|
| 解冻层 | top 6、top 8 | top 6、top 8 | top 10 历史波动更大，先不投入 |
| encoder LR | `5e-6`、`1e-5` | `5e-6`、`1e-5` | `1e-6` 量级历史上欠拟合 |
| head LR | `1e-5` | `1e-5` | 降低额外自由度 |
| seeds | 42 预筛，胜者补 43/44 | 同左 | 控制成本与方差 |

若 top 8/`1e-5` 在 seed42 没有被其他配置提高至少 0.003 AUROC，就直接保留它。三 seed 均值差小于 0.002 时，选择标准差更小、解冻层更少的配置。三 seed mean-logit ensemble 只作为最终交付，不用于选 BERT mask。

## 3. BERT mask 严格消融（最高优先级）

固定 tokenizer checkpoint/cache、factorized CE、batch 128/GPU、LR `3e-4 -> 3e-5`、warmup 2K、50K schedule、per-trial-then-global loss。只改变 mask。

### 3.1 候选

| 编号 | 模式 | ratio | span 分布 | 目的 |
|---|---|---:|---|---|
| M1 | paired random | .50 | 不适用 | 判断连续 span 本身是否有效 |
| M2 | paired random | .60 | 不适用 | random 下难度匹配 |
| M3 | paired span `[1,5]` | .40 | length-uniform | 保留更多局部锚点 |
| M4 | paired span `[1,5]` | .50 | length-uniform | MCI/PD5 折中候选 |
| M5 | paired span `[1,5]` | .60 | length-uniform | 当前基线 |
| M6 | paired span `[1,6]` | .50 | symmetric-power | 复验历史 PD5 候选 |

不再同时扫描 span embedding、`[1,8]`、0.7/0.8 ratio 或复杂多块 JEPA mask。已有证据显示增加这些自由度没有稳定联合收益；先回答 random/span 和难度两个核心问题。

### 3.2 mask 必须满足的不变量

- L/R 在相同时间 patch 同步 mask，阻断另一眼直接泄漏。
- eligibility 为 `L_nonmissing >= 0.85 AND R_nonmissing >= 0.85 AND non-padding`。
- ratio 的分母是 eligibility patch 数，不是 padded sequence 长度。
- stim 不被 mask；stim query 不读取 L/R，CLS/L/R 可读取 stim。
- 每个 trial 先对自己的 masked target loss 求均值，再跨有监督 trial 等权平均。

## 4. 预测目标消融（第二优先级）

使用第 3 节选出的 mask，仅改变监督目标。

| 候选 | 目标 | 优点 | 主要风险 |
|---|---|---|---|
| T1 | FSQ 分维度 CE 之和 | 避免 1575-way 稀疏分类，现有最佳 | 各维独立，忽略 code 维相关性 |
| T2 | 1575-way joint CE | 保留联合 code 分布 | 类别长尾，优化更难 |
| T3 | raw patch reconstruction | 不受 tokenizer 信息瓶颈限制 | 容易偏向低层幅值和简单平滑 |

raw patch 接口按以下定义实现：预测每个 masked 眼 patch 的 `x/y/area/blink`；x/y/area 对有效且非 blink 帧计算 Smooth-L1，blink 对有效帧计算 BCE；missing 帧完全不进入 loss；默认连续项权重 1、blink 权重 0.1。它不加载 tokenizer，也不读取 code-ID cache。若 raw reconstruction 的下游更好，说明 tokenizer code 丢失疾病相关连续信息；若 BERT loss 更低但下游更差，说明低层重建捷径占主导。

## 5. Quantizer 与码本消融（第三优先级）

只在 mask 和目标已经锁定后进行，避免形成不可解释的笛卡尔积。

| 候选 | Codebook | BERT 目标 | 必查质量门槛 |
|---|---:|---|---|
| Q1 标准 FSQ 9755 | 1575 | factorized CE | 正式基线 |
| Q2 iFSQ 9755 | 1575 | factorized CE | PPL、top-1、每维使用率 |
| Q3 FSQ 97755 | 11025 | factorized CE | 稀疏度、第五维是否有效 |
| Q4 VQ-VAE | 1575 learned embeddings | joint CE | dead codes、PPL、commit/codebook loss |

VQ-VAE 使用 4D learned code embedding、1575 个 code、straight-through nearest neighbour、commitment beta 0.25；先进行 1K continuous warmup，再用 K-means 初始化 learned codebook。选择 1575 是为了与 FSQ9755 匹配容量，不把“量化方法”和“code 数量”混在一起。

历史上 97755 只改善过单次 MCI validation，却降低 MCI test、PD5 validation 和 PD5 balanced accuracy；iFSQ code usage 不差但 MCI 表示曾退化。因此 Q2/Q3/Q4 都是机制消融，不应默认认为会超过 Q1。

Tokenizer 候选若出现以下任一条件直接淘汰，不继续训练 BERT：非有限 loss；active code fraction 过低；top-1 code frequency 超过 0.10；PPL 持续下降并趋近 1；验证重建/特征 loss 明显反弹。VQ-VAE 还要求 codebook 参数存在非零梯度，且 dead-code 比例不持续扩大。

## 6. 高效执行顺序

### Stage 0：10K BERT + Train-only 冻结表示探针

- 新 mask 只训练 10K；当前 4-GPU 实测约 40 分钟。
- 冻结 BERT，缓存下游 Train split 的 trial CLS；不读取 validation/test。
- 对每个 subject 先在任务内平均 CLS，再四任务等权平均；使用固定的 subject-stratified 5-fold logistic regression 比较 MCI/PD5 Train OOF AUROC。
- 该阶段只作负向筛选：明显落后的候选淘汰，前两名进入正式筛选。探针第一名不能直接宣布为最终方案，因为它不包含 encoder 微调、demographics 和非线性 head。
- 已有 BERT checkpoint 不需要任何 BERT 训练，直接缓存 CLS 并运行探针即可。

### Stage A：15K BERT + 单 seed 下游筛选

- mask 六组，固定现有 tokenizer/cache。
- BERT 15K，训练 schedule 本身仍按对应阶段配置；用于相对筛选，不把绝对 loss 与 50K 正式模型混报。
- MCI/PD5 各 seed42、最多 50 epochs、跳过 test。
- 每个任务分别保留 top 3；联合候选取两个任务 validation 标准化均值的 top 3。

淘汰门槛：任一任务相对基线下降超过 0.01；出现 NaN；表示冻结探针接近随机；多 checkpoint 下游趋势持续变差。

### Stage B：50K 严格确认

- Stage A top 2 mask 从零按 50K 正式 schedule 训练。
- 固定最佳下游协议，MCI/PD5 seeds 42/43/44。
- 仅依据三 seed validation 均值与标准差选择 mask；冻结选择后才运行 test。

### Stage C：目标比较

- 使用获胜 mask 比较 factorized code、joint code、raw reconstruction。
- 先走同样 15K/seed42 筛选；只有相对基线任一任务提高至少 0.003 且另一任务不下降超过 0.005，才补 50K/三 seed。

### Stage D：量化器比较

- iFSQ、97755、VQ-VAE 先训练 20K tokenizer proxy，通过 code usage gate 后训练 15K BERT proxy。
- 只有联合 validation 分数进入前二，才从零做 tokenizer 40K + BERT 50K + 三 seed 下游。
- 不允许用 test 决定是否晋级。

## 7. 选择指标与诊断

### BERT 训练指标

- code 目标：Val loss、exact joint-code accuracy、每维 CE/accuracy/digit distance。
- raw 目标：valid non-blink continuous Smooth-L1、valid-frame blink BCE/accuracy。
- 所有目标：train-val gap、梯度范数、LR 曲线、masked eligible ratio、15K/30K/40K/50K 表示探针。

Exact code accuracy 只在相同目标和相同 codebook 内可比较，不能拿 raw loss、joint CE 和 factorized CE 的数值直接排序。最终表示质量必须由冻结探针和完整下游 validation 判断。

### 下游主指标

- MCI：三 seed subject-level Val AUROC 均值和 SD。
- PD5：三 seed subject-level Val macro-AUROC OVR 均值和 SD，同时报告每类 AUROC。
- 联合模型：两个任务各自在候选内 z-score 后等权平均；差小于 0.05 SD 时选 worst-task 更高者，再选更简单者。
- Test 只在方案冻结后报告；历史上 test 已多次查看，因此必须标记为内部探索测试。

## 8. 已实现的直接执行接口

- 实验矩阵：`configs/eyevq/ablations/v6_representation_ablation.yaml`
- 统一执行器：`scripts/run_eyevq_representation_ablation.py`
- raw reconstruction：`bert.target_type: raw_patch`
- FSQ 分维度目标：`bert.target_type: factorized_code`
- joint code 目标：`bert.target_type: joint_code`
- iFSQ：`vq.type: fsq` + `vq.fsq_activation: ifsq`
- VQ-VAE：`vq.type: vqvae` + `vq.code_dim/codebook_size/commitment_beta`
- 97755：`vq.fsq_L: [9,7,7,5,5]`

只生成和验证配置，不启动训练：

```bash
PYTHONPATH=src python scripts/run_eyevq_representation_ablation.py --phase probe
```

执行单组快速筛选：

```bash
PYTHONPATH=src python scripts/run_eyevq_representation_ablation.py \
  --phase probe --only mask_random_r050 --execute
```

对 validation 选出的候选做三 seed 正式确认：

```bash
PYTHONPATH=src python scripts/run_eyevq_representation_ablation.py \
  --phase confirm --only mask_span15_r050,target_raw_patch --execute
```

执行器会为每个候选生成相互隔离的 tokenizer/BERT/MCI/PD5 YAML 和 `execution_plan.json`；任一命令失败立即停止。screen 下游带 `--skip-test`，confirm 才生成 test。当前 GPU 1--4 上的 V6 BERT 训练不由该脚本管理，也不会被干跑打断。

## 9. 预计成本与停止条件

当前 BERT 实测约 4.2 steps/s（4 GPU，batch 128/GPU），纯训练 15K 约 1 小时、50K 约 3.3 小时，validation 和下游另计。六组 mask 全部直接跑满并不划算；Stage A 先筛再补三 seed，可将大部分失败候选限制在约一小时 BERT 成本。

若完成严格 random/span/ratio、factorized/joint/raw 三类目标后，三 seed联合 validation 仍没有超过基线至少 0.003，且置信区间高度重叠，则应停止继续堆叠 BERT mask。此时瓶颈更可能是 subject 数量、标签噪声、PD5 类别可分性和 validation 方差；后续资源应转向扩大独立 subject、外部验证、标签审计与任务专用模型，而不是继续扩大 codebook 或 mask 难度。
