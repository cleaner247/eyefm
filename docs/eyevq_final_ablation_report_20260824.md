# EyeVQ 全阶段实验总结与最终方案（2026-08-24）

## 1. 结论

截至本报告日期，完成消融后最适合作为论文主实验的参考方案基于 V4 数据、per-subject/per-eye area median-MAD、标准 FSQ `[9,7,5,5]` Tokenizer 40K；BERT 使用左右眼同步、仅有效 patch 的 `span[1,5]` mask，span 长度均匀，mask ratio 0.6，四维 FSQ 分头 CE，batch 128/GPU，50K steps，LR `3e-4 -> 3e-5`；下游 strict four-task、K=4、shared-head logit mean、raw 16D demographics、hidden 128/dropout 0.3、冻结 embedding 和底部 4 层、顶部 8 层 LR `1e-5`。后续当前 V5 构建错误地对已滤波交付数据再次执行了 75 Hz 滤波，其结果只作为历史探索记录。

选择只依据 validation。历史 test 已被多次观察，所有 test 数字只能标记为内部探索结果。

## 2. 当前训练停止状态

- Top-10/LR `5e-6` 流水线已按用户要求停止。
- MCI 三个 seed 已完成；PD5 尚未完成，不能用于 Top-8/Top-10 的双任务比较。
- GPU 1--4 已释放。
- 排队器中“僵尸进程仍阻塞后续流水线”的错误已修复。

## 3. 数据和 Tokenizer 消融

### 3.1 数据版本

同类旧流水线的单 seed 结果：

| 数据 | MCI Val/Test AUC | PD5 Val/Test AUC | 判断 |
|---|---:|---:|---|
| V3 | 0.8801 / 0.8667 | 0.8812 / 0.8747 | 高频保留较多，BERT target 更难 |
| V3.1 | 0.8731 / 0.8889 | 0.8645 / 0.8785 | 完全不滤高频，没有稳定收益 |
| V4 | **0.8931 / 0.9018** | 0.8718 / **0.8837** | 统一方案最均衡 |

结论：该表支持 V4 在当时三套清洗数据中最均衡。当前 V5 的输入本已滤波，但生成器又执行了一次保护式 75 Hz 零相位滤波，因而不是有效的数据消融候选。正确 V5 应保持交付坐标不变，只重建标签、特征、索引和 packed 表示。

### 3.2 FSQ

| FSQ | codes | MCI Val/Test | PD5 Val/Test | 结论 |
|---|---:|---:|---:|---|
| `[9,7,5,5]` | 1,575 | 0.8790 / **0.9004** | **0.8854** / 0.8733 | 稳定默认 |
| `[9,7,7,5,5]` | 11,025 | **0.9107** / 0.8875 | 0.8671 / 0.8799 | 更稀疏，收益不一致 |

标准 FSQ 9755 比 iFSQ 9755 的 V3 下游明显更好；iFSQ 只证明能防硬坍缩，没有证明表示更优。

### 3.3 Tokenizer 学习率与长度

- 短程 LR 扫描中 `3e-4` 的 L_eye/L_feat/code usage 综合优于 `2e-4` 和 `5e-4`。
- warmup 2K 已足够；5K 没有收益。
- velocity 是 XY 的派生目标，weight 0 最稳定。
- 40K checkpoint 指标持续优于 25K/30K/35K，且码本没有坍缩：

| Step | L_eye | L_feat | PPL | Active | Top-1 |
|---:|---:|---:|---:|---:|---:|
| 25K | 0.001604 | 0.16483 | 736.5 | 1405 | 0.0173 |
| 30K | 0.001573 | 0.15725 | 780.1 | 1410 | 0.0151 |
| 35K | 0.001529 | 0.15172 | 794.7 | 1452 | **0.0140** |
| **40K** | **0.001495** | **0.15119** | **798.2** | **1469** | 0.0168 |

40K 相比 35K 的 L_feat 改善已很小，但 L_eye、PPL、active codes 仍改善；因此选 40K，不继续延长。

正式 loss：XY 1.0、area 0.1、blink 0.1、velocity 0；manual group 0.0015、binary 0.25、continuous 1.5，count 不参与 loss。

## 4. BERT mask 消融

### 4.1 已完成三种子、可比较结果

| Mask/预测方式 | BERT | MCI Val/Test | PD5 Val/Test | 判断 |
|---|---|---:|---:|---|
| span2--6, ratio .25 | joint CE, 40K | 0.8884 / 0.8805 | 0.8606 / 0.8719 | mask 偏容易 |
| span2--6, ratio .50 | joint CE, 40K | 0.8982 / 0.8834 | 0.8608 / 0.8684 | MCI 改善，PD5 不改善 |
| span1--6, symmetric, ratio .40 | joint CE | 0.8825 / 0.8815 | 未完成 | 证据不全 |
| span1--6, symmetric, ratio .50 | joint CE | 0.8672 / 0.8724 | **0.8687** / 0.8699 | PD5 较稳，MCI 较差 |
| span1--6, symmetric, ratio .60 | joint CE | **0.9146** / 0.8875 | 0.8586 / 0.8699 | MCI 最强、PD5 下降 |
| span1--6, token-balanced, ratio .60 | joint CE, 50K | 0.8892 / 0.8896 | 0.8560 / 0.8681 | 不如简单均匀策略 |
| span1--8, uniform, ratio .60 | factorized CE, 80K | 0.8880 / 0.8951 | 0.8664 / 0.8663 | 长 span/80K 无稳定收益 |
| **span1--5, uniform, ratio .60** | **factorized CE, 50K** | **0.9017 / 0.8810** | 0.8626 / **0.8810** | 最佳简单统一方案 |
| span1--5, token-balanced + span embedding | factorized CE, B256/40K | 0.8664 / **0.9030** | **0.8688** / 0.8754 | Val/Test 反向，不按 Test 选 |

注意：最后一组同时改变了 batch、LR、steps、span embedding 和分布，不能把差异单独归因于 token-balanced。它的 validation 联合表现没有超过简单 span1--5 uniform，因此不作为主方案。

### 4.2 mask ratio 判断

- 旧 V4 iFSQ 的严格 ratio 0.15/0.25 对照：0.15 提高 BERT accuracy、MCI Val 和 PD5 Val，但两个 Test AUC 略低。
- 新 span 实验中 ratio 0.6 对 MCI 学习长程依赖明显有效；ratio 0.4/0.5 对 PD5 更温和。
- 两任务存在真实冲突：MCI 偏好更强、更连续的遮挡；PD5 更依赖局部可辨别动态。
- 为保持一个统一模型，选择 ratio 0.6 + span1--5；若允许任务专用预训练，MCI 可用 0.6，PD5 应在 0.5/0.6 中按三种子 Val 复验。

### 4.3 为什么不保留 span-length embedding

它给预测器直接提供任务难度/遮挡尺度信息，可能帮助 MLM loss，却也可能让模型依赖 mask 元数据而不是上下文。现有带 embedding 的运行没有提高双任务 validation 联合表现，因此主模型删除该机制，保持架构简洁。

## 5. 下游消融

### 5.1 聚合与样本策略

- K4 相比 K2：旧受控实验 Val AUC `+0.0076`、Test AUC `+0.0122`，且只少 1 个训练 subject；选择 K4。
- strict four-task 比 partial-task 更稳定。partial 虽恢复 PD5 58 个训练 subject，但 Val/Test AUC 均下降。
- shared trial logits mean 比 CLS 笛卡尔拼接、残差融合和 mask-CLS partial 方案更简单且更稳。
- demographics 使用训练集拟合的原始 16D 编码直接拼接 `LayerNorm(CLS)`；升维到 128、投影到 4 后再联合 LayerNorm 都没有稳定收益。
- hidden sweep 中 128 的 PD5 Val macro-AUC 0.8882，高于 256/64/32/16；选择 hidden 128、dropout 0.3。

### 5.2 解冻层数

当前同 BERT、同下游配置的三种子 MCI：

| 配置 | Val AUC | Test AUC | Val BAcc | Test BAcc |
|---|---:|---:|---:|---:|
| **Top-8, LR 1e-5** | 0.8664 +/- 0.0095 | **0.9030 +/- 0.0076** | **0.8092** | **0.8116** |
| Top-10, LR 5e-6 | **0.8739 +/- 0.0096** | 0.8872 +/- 0.0094 | 0.7840 | 0.7951 |

Top-10 的 Val AUC 高 0.0074，但 Val BAcc、Test AUC、Test BAcc 全部下降，而且最佳 epoch 更晚。结合旧 Top-8/Top-10 单 seed 对照（Top-10 Val 仅 +0.0012、Test -0.0079），Top-8 是更稳健的正式选择。Top-10 的 PD5 实验未完成，不作结论。

### 5.3 正则化与训练控制

- consistency 0.05 曾在单 seed MCI 带来 Val +0.0024/Test +0.0057，但没有三种子确认，主方案保持 0。
- 100 epochs 是上限，不是固定用满；raw Val AUROC 最早 epoch 27 后 patience 20。
- 每次新 best 必须清零 patience；结束或早停后加载 `ckpt_best.pt` 再测试。
- MCI 使用 subject positive weight、阈值固定 0.5；PD5 使用 subject inverse-frequency class weight、argmax。

## 6. 最终训练配置

### Tokenizer

- 12x384 encoder、3x384 decoder、8 heads、FFN 1152；patch 40/stride 40。
- stim-isolated attention；标准 FSQ `[9,7,5,5]`。
- batch 128/GPU x 4，bf16，AdamW beta `(0.9,0.95)`，matrix WD 0.01。
- LR `3e-4 -> 3e-5`，warmup 2K，cosine，总 40K；val/save 2.5K。

### BERT

- 12x384、8 heads、FFN 1152、dropout 0。
- L/R 同步 mask；只在 nonmissing >= 0.85 且非 padding 的 patch 中计算 0.6 比例。
- span `[1,5]`，每个 span 长度等概率；不使用 span embedding。
- 四个 FSQ 维度分别 CE 后求和；每 trial 先按有效 masked target 平均，再跨 trial 平均。
- batch 128/GPU x 4，50K；LR `3e-4 -> 3e-5`，warmup 2K，cosine。

### MCI/PD5

- strict four-task；K4 无放回；eval 使用所有有效 trials。
- `LayerNorm(CLS384) + raw demographics16 -> MLP(400,128,out)`，GELU，dropout .3。
- 每任务先平均 trial logits，再等权平均四任务 logits。
- embedding/MLM head 冻结，底部 4 层冻结，顶部 8 层统一 LR `1e-5`；head LR `1e-5`。
- subjects/GPU 4 x 4；100 epochs 上限，warmup 4，cosine 到 0.1 倍；Val AUC early stop。

## 7. 各阶段最终消融确认矩阵

以下矩阵用于论文表格。已经完成的结果直接引用；缺失项若补实验，必须一次只改变一个变量，且只按三种子 validation 排序。

| 阶段 | 主配置 | 对照 | 当前状态 |
|---|---|---|---|
| 数据 | V4 | V3、V3.1 | 已完成，B级 |
| area | per-subject/per-eye MAD | pooled-eye/global | 旧实现混杂；只作历史参考 |
| FSQ | 9755 | iFSQ9755、97755 | 已完成，B级 |
| Tokenizer LR | 3e-4 | 2e-4、5e-4 | 已完成短程 A级 |
| Tokenizer step | 40K | 25/30/35K | 重建指标完成；下游代理未全完成 |
| velocity | 0 | 0.05/0.1 | 只有训练行为证据，未做完整下游 A级消融 |
| BERT prediction | factorized CE | joint 1575-way CE | 现有结果有其他变量混杂，尚缺严格 A级 |
| mask span | uniform 1--5 | 1--6、1--8、random | 三种子结果已完成但训练量部分不同 |
| mask ratio | 0.6 | 0.4、0.5 | span1--6 已完成；span1--5 严格对照未完成 |
| span embedding | off | on | on 组与 batch/LR 同时变化，尚缺严格 A级 |
| K | 4 | 2 | 已完成 A级 |
| task completeness | strict | partial | 已完成 A级 |
| hidden | 128 | 16/32/64/256 | 已完成单 seed |
| unfreeze | top 8 | top 6/top 10 | MCI 已有；PD5 top10 未完成 |
| consistency | 0 | 0.05 | 单 seed完成，三种子未确认 |

若只允许补最小实验，应按优先级运行：

1. 固定 span1--5/factorized/B128/50K，只比较 ratio 0.5 与 0.6。
2. 固定上述胜者，只比较 factorized CE 与 joint CE。
3. 固定上述胜者，只比较 span embedding on/off。
4. 用最终 BERT 对 PD5 比较 top-6/top-8；不再优先投入 top-10。

每个候选先 seed 42 快筛；只有 Val 提升超过 0.005 才补 seed 43/44。最终采用三 seed Val 均值；Test 不参与筛选。

## 8. 最终判断

当前瓶颈不是 Tokenizer 码本容量或 BERT 是否继续训练，而是：固定小规模 subject split 的方差、MCI/PD5 对 mask 尺度的偏好冲突，以及 PD5 的类别决策校准。继续增加 span、mask ratio、解冻层数和架构条件信息会增加自由度，却没有形成一致的双任务 validation 收益。正式论文应采用上述简单统一方案，并把任务专用 mask 作为补充实验，而不是继续堆叠机制。
