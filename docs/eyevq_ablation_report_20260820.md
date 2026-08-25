# EyeVQ 全部消融实验汇总报告

生成日期：2026-08-20
项目：`/mnt/disk_sde/hjf/pxy_eyefm`
结果范围：以 `outputs/eyevq/` 下具有可追溯日志、配置或 `metrics_test.json` 的 EyeVQ 实验为主。

## 1. 摘要结论

目前证据最支持的统一方案是：

- 数据使用 **V4**；V4 的高频清洗显著降低 Tokenizer 的重建难度，同时是三套同配置流水线中 MCI 验证/测试最好的数据版本。
- 量化使用标准 **FSQ `[9,7,5,5]`**。`[9,7,7,5,5]` 只提高过单次 MCI validation，未在 MCI test、PD5 validation 和 balanced accuracy 上形成一致收益。
- Tokenizer 使用新 loss 配置、40K steps、`3e-4 → 3e-5` cosine、warmup 2K、velocity weight 0。V4 40K 最终达到 `L_eye=0.00032`、`L_feat=0.10839`、active codes 1407、PPL 745，未发生码本坍缩。
- BERT mask ratio 优先 **0.15**。与 0.25 的严格对照中，0.15 明显提高 BERT validation accuracy，并提高 MCI validation AUC 和 PD5 balanced accuracy；但两任务 test AUC 都略低，因此结论是“更适合当前验证选择”，不是全面碾压。
- MCI/PD5 下游采用 strict four-task、K=4、trial logits 先按任务平均、四任务等权平均、共享 MLP head、hidden 128、dropout 0.3、解冻顶部 8 层、embedding 冻结、encoder/head LR 都为 `1e-5`。
- 人口学信息采用训练集拟合的 16D 编码，其中年龄为一维 z-score；对每个 `LayerNorm(CLS)` 直接拼接。年龄不归一化在 validation 上明显变差。
- partial-task 能恢复数据，尤其 PD5 训练集从 514 增至 572 subjects，但没有提高 AUC；当前最终方案应保持 strict four-task。
- 多视图 consistency weight 0.05 对 MCI 有小幅正收益，但增益约 0.002–0.006 AUC、置信区间跨零；不属于稳定的核心收益，统一正式基线仍可设为 0。
- 本项目历史 test 已被多次查看，因此本文 test 只能作为内部探索结果，不能当作完全未见的外部测试。

当前正在运行的 `v4_fsq9755_nodecay_optimal_search` 尚未完成，不纳入胜负排序。截至报告生成时只完成到 Tokenizer 约 5K step。

## 2. 统计口径和可比性

### 2.1 主要指标

- MCI：subject-level validation/test AUROC；balanced accuracy 使用固定阈值 0.5 时优先报告固定阈值结果。
- PD5：subject-level macro one-vs-rest AUROC、balanced accuracy，预测为 softmax argmax。
- Tokenizer：validation `L_eye`、`L_feat`、XY/area/blink loss、PPL、active codes、top-1 code frequency。
- BERT：validation MLM loss、top-1 accuracy、top-5/top-10 accuracy。

### 2.2 证据等级

- **A级：严格可比**——同数据、同 checkpoint、同下游架构，只改变一个变量。
- **B级：基本可比**——总体流程相同，但 tokenizer step、数据清洗或个别 head 细节同时变化。
- **C级：历史参考**——旧数据统计、旧采样、旧阈值、代码缺陷或缺少 test；不用于最终选型。

注意：较低的 Tokenizer/BERT loss 不自动代表更好的下游表示。数据清洗会改变任务难度和 label entropy，因此最终仍以固定下游 validation 为选择依据。

## 3. 数据与归一化消融

### 3.1 V3、V3.1、V4 同配置流水线（B级）

共同设置：标准 FSQ 9755、新 loss、Tokenizer 40K、BERT 50K、mask 0.15、下游 K4/top-8/shared head hidden128/dropout0.3、seed 42。

| 数据 | Tok `L_eye` | Tok `L_feat` | PPL / active | BERT loss / acc | MCI Val AUC | MCI Test AUC | MCI Test BAcc@0.5 | PD5 Val AUC | PD5 Test AUC | PD5 Test BAcc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V3 | 0.00149 | 0.13182 | 771 / 1432 | 1.605 / 0.480 | 0.8801 | 0.8667 | 0.7910 | 0.8812 | 0.8747 | 0.4392 |
| V3.1（不滤高频） | 0.00151 | 0.15387 | 804 / 1490 | 1.647 / 0.467 | 0.8731 | 0.8889 | 0.8032 | 0.8645 | 0.8785 | 0.3870 |
| **V4（全滤高频）** | **0.00032** | **0.10839** | 745 / 1407 | **0.859 / 0.683** | **0.8931** | **0.9018** | 0.7971 | 0.8718 | **0.8837** | 0.4326 |

判断：

- V4 将 area loss 从 V3/V3.1 的约 0.013 降至 0.00147，是 `L_eye` 大幅下降的主要来源之一。
- V4 的 BERT token 更容易预测，accuracy 从约 0.47–0.48 提升至 0.683。
- V4 对 MCI 是明确收益；PD5 validation 不如 V3，但 test AUC 最好，balanced accuracy 居中。
- V3.1 虽然拥有最高 code usage，但 BERT 和下游没有同步变好，说明保留全部高频信息增加了不可预测噪声。
- 数据版本的最终选择应为 V4，但需要承认 V4 可能让重建目标更平滑；不能单凭重建 loss 宣称它保留了更多生理信息。

### 3.2 area normalization 的历史结论（C级到B级）

实验演进为：全局统计 → per-subject pooled-eye → per-subject/per-eye median-MAD。最终 per-eye 形式修复了左右眼尺度差异，也避免了无效眼进入统计。

历史上使用旧 per-subject area-stat 曾出现 MCI Val AUC 接近或超过 0.9，但后续审计发现输入数据、下游数据和统计文件并未始终严格对齐。因此旧高分不能证明“旧统计更好”，只能说明 area scaling 对下游很敏感。

正式规则：

- 左右眼独立统计；median/MAD，MAD scale 1.4826，clip 5，`log1p=false`。
- 双眼均无效的 trial 在 Tokenizer、BERT、finetune 都过滤。
- 单眼无效时，仅有效眼参与重建、code target 和 mask；无效眼 token 不参与 loss。
- 下游允许 transductive 的无标签 subject area-stat，但不能用 label 或 test metric 参与统计/选择。

## 4. Tokenizer 消融

### 4.1 学习率扫描（A级，iFSQ 短程 3K）

固定 batch 128/GPU、velocity=0、其余 loss 相同。

| LR | Val `L_eye` | Val `L_feat` | PPL | Active | top-1 | 梯度状态 |
|---:|---:|---:|---:|---:|---:|---|
| 2e-4 | 0.001888 | 0.3699 | 359.5 | 1294 | 0.0181 | 稳定、无裁剪 |
| **3e-4** | **0.001721** | **0.3393** | **388.8** | **1325** | 0.0182 | 稳定、无裁剪 |
| 5e-4 | 0.001841 | 0.3503 | 376.7 | 1246 | 0.0244 | 稳定、但无额外收益 |

结论：3e-4 是短程综合最优；5e-4 没有更快或更低的 validation loss，且 code usage 略差。正式使用 `3e-4 → 3e-5`。

### 4.2 warmup / AE 初始化 / iFSQ

- 5K warmup 被认为过长；2K warmup 足以将 raw gradient 降到 clip 1 以下。
- AE-only 初始化增加流程复杂度，且没有形成可复现的下游收益。
- iFSQ 能防止“PPL=1”的硬坍缩，但 V3 同配置的最终下游明显不如标准 FSQ，因此防坍缩不能等同于表示质量提高。
- 最终选择标准 tanh FSQ + 2K warmup，不使用 AE 初始化。

### 4.3 velocity loss

可用的受控扫描并不完整：velocity=0.1 的续训在约 1.5K 尚未完成，不能作为完整 A/B 结果。不过训练行为和定义支持以下判断：

- velocity 是相邻 XY 的派生量，XY 收敛后 velocity loss 会自然下降。
- 在早期 XY 误差较大时放大 velocity 会对相邻差分噪声施加强梯度，增加梯度裁剪和码本不稳定风险。
- 现有完整流水线在 velocity=0 时获得了最好的 V4 Tokenizer/BERT/MCI 组合结果。

因此当前正式值为 `eye_velocity_weight=0`。若以后重测，必须从同一 checkpoint 分叉、相同 LR schedule，至少比较 0、0.05、0.1 并完成下游，而不是仅看短程 velocity loss。

### 4.4 loss 重配与训练长度（B级）

旧配置中 manual feature binary loss 在约 30K 后上升，`L_feat` 出现过拟合，而 XY/area 仍继续下降。30K 续训时将 area weight 降至 0.1、binary BCE 再乘 0.5、manual group 降低后，V4 最终得到：

| 配置/阶段 | Val `L_eye` | Val `L_feat` | Binary | Continuous | Area | 说明 |
|---|---:|---:|---:|---:|---:|---|
| 旧 iFSQ 后段约 40–45K | 0.00057–0.00059 | 0.123–0.186 | 持续上升 | 约 0.062 | 约 0.0014–0.0017 | binary feature 过拟合明显 |
| 新标准 FSQ V4 40K | **0.00032** | **0.10839** | **0.02212** | 0.08627 | **0.00147** | 综合最好、code usage 正常 |

推荐 loss：

```yaml
eye_xy_weight: 1.0
eye_area_weight: 0.1
eye_blink_weight: 0.1
eye_blink_pos_weight: 1.0
eye_velocity_weight: 0.0
eye_recon_group_weight: 1.0
eye_commit_group_weight: 0.0
manual_feature_balanced_bce: false
manual_feature_group_weight: 0.0015
manual_feature_binary_weight: 0.25
manual_feature_continuous_weight: 1.5
```

count features 不参与 loss；binary 用 BCE，连续 manual features、XY、area 使用 SmoothL1。现有结果不支持把所有连续项改为 MSE：MSE 会进一步放大少数异常点，而 V4 的目标正是降低高频/异常值对表示的支配。

### 4.5 Tokenizer checkpoint

早期 V4 iFSQ 轨迹显示：

- 30K 左右 continuous feature 尚在改善；
- 32.5K 后 binary feature 开始明显上升；
- 40–45K 的 XY/area 更低，但旧 loss 下 `L_feat` 更差；
- 用重配 loss 从 30K 续训能降低总 validation loss，但改变 loss 后旧、新 checkpoint 的 `val/loss` 不能直接比较。

当前证据只能支持“40K 是新 loss 下合理终点”，尚不能证明 25K/30K/35K/40K 中哪个下游最优。正在运行的 optimal-search 会用每个 tokenizer checkpoint 独立 cache、BERT proxy 和双下游 validation 完成这个缺口。

## 5. FSQ 与码本规模消融

### 5.1 标准 FSQ 9755 vs iFSQ 9755（V3，B级）

| 量化 | MCI Val AUC | MCI Test AUC | PD5 Val AUC | PD5 Test AUC | PD5 Test BAcc |
|---|---:|---:|---:|---:|---:|
| **标准 FSQ 9755** | **0.8801** | **0.8667** | **0.8812** | **0.8747** | **0.4392** |
| iFSQ 9755 | 0.8049 | 0.8143 | 0.8667 | 0.8712 | 0.4112 |

iFSQ 的 code usage 不差，但 MCI 表示明显退化。结论：iFSQ 可作为防硬坍缩手段，不应作为正式默认量化。

### 5.2 9755 vs 97755（B级）

早期 per-subject/per-eye BERT50K 下游：

| FSQ | codes | MCI Val/Test AUC | PD5 Val/Test AUC | PD5 Test BAcc |
|---|---:|---:|---:|---:|
| **[9,7,5,5]** | 1575 | 0.8790 / **0.9004** | **0.8854** / 0.8733 | **0.5017** |
| [9,7,7,5,5] | 11025 | **0.9107** / 0.8875 | 0.8671 / **0.8799** | 0.3993 |

97755 提高了单次 MCI validation，却降低 MCI test、PD5 validation 和 PD5 balanced accuracy。更大离散空间还显著增加 MLM 分类难度及稀疏 code 风险。跨两个任务看，9755 更稳定，作为统一码本更合理。

## 6. BERT 消融

### 6.1 mask ratio 0.15 vs 0.25（A级）

共同使用 V4 iFSQ 9755 tokenizer step45K、同一 code cache、BERT 50K、seed 42。

| Mask | BERT Val loss | Acc | Top-5 | MCI Val/Test AUC | MCI BAcc@0.5 | PD5 Val/Test AUC | PD5 BAcc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.8838 | 0.6827 | 0.9675 | 0.8590 / **0.8875** | **0.7860** | 0.8736 / **0.8836** | 0.4503 |
| **0.15** | **0.7994** | **0.7079** | **0.9747** | **0.8907** / 0.8832 | 0.7810 | **0.8762** / 0.8762 | **0.5001** |

结论：

- 0.15 更容易优化，MCI validation `+0.0317`，PD5 validation `+0.0027`，PD5 BAcc `+0.0498`。
- 0.25 的两个 test AUC 略高，说明 0.15 的收益不是所有指标一致；单 seed 下差异可能含抽样噪声。
- 按只使用 validation 的规则，当前选择 0.15。
- 0.35 实验被停止，未形成完整下游结果，不能声称它能学到“更深相关性”。

### 6.2 BERT 数据版本影响

V4 BERT validation accuracy 0.683，显著高于 V3 0.480、V3.1 0.467。结合相同 mask 0.15，这一差异主要来自 tokenizer target/data entropy，而不是训练实现差异。

但是 BERT accuracy 与下游并非单调对应：V3 的 PD5 Val AUC 仍高于 V4。因此 BERT loss/accuracy 只能做质量门槛，不能直接替代下游 checkpoint 选择。

### 6.3 BERT checkpoint / 训练量

现有完整流水线多数只用 50K final 做下游；30K、40K、50K 的双任务受控比较尚未完成。现有曲线表明 50K 时 LR 已正确余弦下降到约 `5e-5`，loss 仍缓慢下降，没有明显发散。

因此当前 50K 是安全默认值，但“50K 一定优于 30K/40K”仍是待 optimal-search 验证的假设。

## 7. MCI 微调消融

### 7.1 学习率与 epoch（C级历史参考）

早期 batch16 trial/subject pipeline：

| 配置 | Test subject AUC |
|---|---:|
| LR 1e-5，5 epochs | 0.8633 |
| **LR 1e-5，10 epochs** | **0.8710** |
| LR 2e-5，10 epochs | 0.8668 |
| encoder 1e-6 / head 2e-5，10 epochs | 0.8100 |

早期实验已显示 1e-6 过小、2e-6/5e-6 普遍不如 1e-5。后来 subject-MIL 训练中最佳 epoch 常在 30–65，证明 5 或 10 epochs 对当前架构通常不足。正式设置 100 epochs + validation AUROC early stopping 是合理的。

### 7.2 K=2 vs K=4（A级，旧 concat head）

| K | Train subjects | Val AUC | Test AUC | Test BAcc | Test loss |
|---:|---:|---:|---:|---:|---:|
| 2 | 242 | 0.8079 | 0.7602 | 0.7437 | **0.6182** |
| **4** | 241 | **0.8155** | **0.7724** | **0.7437** | 0.6433 |

K4 对 AUC 有小幅一致收益；K2 并没有体现出更强、有效的正则化。K4 仅少保留 1 个训练 subject，因此数据损失很小。正式使用 K4，训练无放回采样，validation/test 使用全部有效 trial。

### 7.3 partial-task vs strict four-task（A级）

| 任务 | 策略 | Train subjects | Val subjects | Test subjects | Val AUC | Test AUC | Test BAcc |
|---|---|---:|---:|---:|---:|---:|---:|
| MCI | **strict** | 241 | 60 | 76 | **0.9013** | **0.9032** | **0.8466@0.5** |
| MCI | partial, unweighted | 245 | 61 | 77 | 0.8998 | 0.8829 | 0.7724@0.5 |
| PD5 | **strict** | 514 | 124 | 156 | **0.8882** | **0.8843** | **0.5114** |
| PD5 | partial, unweighted | 572 | 143 | 178 | 0.8800 | 0.8707 | 0.4943 |

partial-task 恢复的数据：MCI 仅 `+4/+1/+1` subjects；PD5 为 `+58/+19/+22`。尽管 PD5 恢复较多数据，质量和任务覆盖差异抵消了样本量收益。正式选 strict。

此前尝试按缺失任务数给 loss 加权没有形成优于 unweighted 的证据；partial 内部应保持不加权，但最终方案不启用 partial。

### 7.4 解冻层数

较早的 120-epoch validation-only 搜索曾得到 6/8/10 层 Val AUC 0.912/0.929/0.925，12 层全解冻末期仅 0.744；但该组没有最终 test，且使用旧数据/旧评估流程，只作为“全解冻不稳定”的历史证据。

在修复后、同一 shared-head + consistency 0.05 的严格对照中：

| 解冻顶部层数 | Val AUC | Test AUC | 备注 |
|---:|---:|---:|---|
| 6 | 0.8837 | 0.8932 | 偏保守 |
| **8** | 0.9001 | **0.8982** | 最均衡 |
| 10 | **0.9013** | 0.8903 | Val 仅 +0.0012，Test -0.0079，波动更大 |

结论：8 层是稳定默认；10 层可能在某个 seed 提高 validation，但没有稳定泛化收益。12 层/全量微调更容易过拟合，embedding 继续冻结。

### 7.5 consistency weight（A级）

在 top-8、encoder/head LR `1e-5` 下：

| Weight | Val AUC | Test AUC | Test BAcc |
|---:|---:|---:|---:|
| 0 | 0.8978 | 0.8925 | 0.7971 |
| **0.05** | **0.9001** | **0.8982** | **0.8082** |

在 top-6、head LR `2e-5` 下：

| Weight | Val AUC | Test AUC |
|---:|---:|---:|
| 0.02 | 0.8801 | **0.8932** |
| 0.05 | 0.8801 | **0.8932** |
| 0.10 | **0.8837** | 0.8853 |

0.05 相对 0 有小幅一致收益；0.10 更像提高单次 validation 而牺牲 test。由于差值小且只有 seed42，正式统一基线可保持 0，以减少目标耦合；若专门优化 MCI，可将 0.05 作为次选并进行多 seed 复验。

### 7.6 task/trial 聚合架构

后期同一强基线附近的结果：

| 架构 | Val AUC | Test AUC | Test BAcc@0.5 | 判断 |
|---|---:|---:|---:|---|
| **shared trial-logit mean，strict** | **0.9013** | **0.9032** | **0.8466** | 最稳定 |
| 4-task Cartesian CLS concat，hidden64 | 0.8884 | 0.8803 | 0.8194 | 组合数大、较易过拟合 |
| shared logits + residual fusion h16，strict | 0.8848 | 0.9018 | 0.8032 | test AUC 高但 validation 较差 |
| residual fusion h16，partial | 0.8863 | 0.8892 | 0.7567 | 无收益 |
| Cartesian + mask CLS，partial | 0.9009 | 0.8815 | 0.7998 | validation 高但 test 不稳 |

早期低性能 concat-head 对照中，concat h32 比 shared h32 仅高约 0.007 Val/Test AUC；在修复 area-stat、有效眼过滤、人口学融合和 trial-logit mean 后，shared head 明显更好。因此不能用早期小优势否定后期 shared-head 结论。

“先枚举 K^4 个四任务 CLS 拼接再平均 logits”会扩大高度相关的组合数，却没有增加独立 subject 信息；validation 时组合可达上万，容易造成伪精确和过拟合。现有结果支持每个 trial 独立过共享 head、先任务内平均、再任务间平均。

### 7.7 人口学编码与分类头

年龄 raw vs train-fold z-score（A级）：

| 年龄编码 | Val AUC | Test AUC | Val loss | Test loss |
|---|---:|---:|---:|---:|
| raw years | 0.8778 | 0.9018 | 0.4732 | 0.4435 |
| **一维 z-score** | **0.9013** | **0.9032** | **0.4011** | **0.4204** |

z-score 的 validation AUC `+0.0235`，虽 bootstrap 95% CI 跨零，但 loss/Brier 也同步改善，因此比 raw age 更合理。

人口学 head 形式：

| Head | Val AUC | Test AUC |
|---|---:|---:|
| linear | 0.8754 | 0.8695 |
| MLP hidden32 | 0.8702 | **0.8946** |
| **MLP hidden128** | **0.8801** | 0.8932 |

这里的 validation 选择支持 hidden128。将 16D demographics 升维到 128 后再与 CLS 共同 LayerNorm 的方案容易让人口学分支占比过大；`Linear(16,4)` 的实验没有完成完整 PD5/test，因此不能作为胜者。当前最稳的是只对 384D CLS 做 LayerNorm，再直接拼接 16D demographics，输入总维 400。

### 7.8 其他正则化

旧 concat h32 基线中：

- auxiliary task loss 0.1：Val/Test AUC 0.803/0.741；没有收益。
- K2×2 consistency 0.1：0.805/0.741；没有收益。
- mixup alpha 0.4：0.817/0.777，是该旧基线中最好，但仍明显低于后期 shared-head 0.90 级结果。
- bottleneck16/top4/encoder LR 2e-6：0.726/0.673，明显欠拟合。

这些实验说明修复数据和聚合机制的收益远大于在弱基线上叠加 auxiliary/mixup/bottleneck。

### 7.9 subject batch、采样和训练计数

这一组经历过 `subjects_per_gpu={1,4,8,16}`、step-based balanced quota、近自然比例 quota 和 epoch shuffle，多数不是只改变单变量的完整 A/B，因此不报告伪精确差值。可以确认的工程结论是：

| 方案 | 观察 | 最终判断 |
|---|---|---|
| 1 subject/GPU | 梯度噪声最大，单步类别组成极端，吞吐低 | 不采用 |
| 4 subjects/GPU | 显存、subject 多样性和 step 数较均衡 | **采用** |
| 8–16 subjects/GPU | 有效 batch 变大，但每 epoch optimizer step 变少；小数据下更新次数不足 | 不作为默认 |
| 每 step 强制正负完全均衡 | 少数类 subject 被高频重复，训练分布偏离真实比例 | 删除 |
| 近自然比例配额 | 比完全均衡温和，但仍引入人为重复 | 后来删除 |
| epoch 内 subject shuffle、无 class quota | 每个 subject 每 epoch 最多一次，分布最自然 | **采用** |

当 batch 不再强制均衡时，MCI 恢复 subject positive weight，PD5 使用 inverse-frequency class weight；不会同时再加 sampler reweighting，避免双重校正。

### 7.10 task weight、layer-wise LR decay 与早停

- learnable task softmax weight 在早期训练中几乎不动。四任务梯度和表征相近，且 softmax 初始对称；在小 subject 数据上增加 task-weight 参数没有稳定收益，因此改为四任务固定等权。
- task-weight entropy regularization 设为 0 后也没有显示出需要恢复的证据。
- layer-wise LR decay 0.8/0.9 与“冻结底层、解冻顶部若干层同 LR”没有完成严格同条件比较；后者更容易解释且在 top-8 上稳定，因此正式关闭 layer-wise decay。
- head LR `2e-5` 在部分 top-6/multiview 实验中可训练，但没有优于 encoder/head 都为 `1e-5` 的强基线；最终统一为 `1e-5`。
- 按 step 训练 1000/2000 步会重复采样 subject，且与数据集规模脱钩；最终恢复 epoch 计数。
- 滑动平均 AUC 曾有 new-best 后 patience 未清零的实现错误。修复后仍发现滑动指标会滞后真实峰值，因此正式使用 raw validation AUC；每次 new best 必须将 patience 清零，early stop 和正常结束都加载 `ckpt_best.pt` 后测试。

### 7.11 attention 与 mask 机制：作为不变量保留

这部分在排除信息泄漏后不再作为自由超参数：

- stimulus query 不允许读取 L/R token；L/R 和 CLS 可以读取 stimulus。
- BERT 在同一时间 patch 成对 mask L/R，避免另一眼直接泄漏同步目标。
- 只有非 padding、有效眼且 `nonmissing >= 0.85` 的 token 能成为 mask target。
- finetune 沿用相同结构性 attention mask；MLM head 冻结并丢弃。

上述约束的目的不是提高某个单次 AUC，而是避免目标泄漏并保证预训练、微调语义一致。

## 8. PD5 微调消融

### 8.1 hidden dimension（A级）

固定 strict、K4、top-8、dropout0.3、LR1e-5：

| Hidden | Val macro-AUC | Test macro-AUC | Test BAcc |
|---:|---:|---:|---:|
| 16 | 0.8410 | 0.8604 | 0.4728 |
| 32 | 0.8737 | 0.8724 | 0.4078 |
| 64 | 0.8719 | 0.8794 | 0.4160 |
| **128** | **0.8882** | 0.8843 | **0.5114** |
| 256 | 0.8687 | **0.8846** | 0.4624 |

hidden128 是 validation 与 balanced accuracy 的明确胜者。256 只在 test AUC 上比 128 高 0.0003，不应据 test 反选。

### 8.2 dropout（A级）

固定 hidden128：

| Dropout | Val AUC | Test AUC | Test BAcc |
|---:|---:|---:|---:|
| **0.3** | **0.8882** | **0.8843** | **0.5114** |
| 0.2 | 0.8798 | 0.8753 | 0.4953 |

0.3 在所有主要指标上更好，PD5 小类别任务需要更强正则。

### 8.3 PD5 主要误差来源

strict 最佳结果的 macro-AUC 约 0.884，但 balanced accuracy 仅约 0.51。说明排序能力尚可，argmax 分类边界和少数类召回仍弱。partial-task 增加 58 个训练 subject 也未改善，问题不只是样本数，而是小类别特征重叠、任务覆盖和类别不平衡。

当前 inverse-frequency class weight 是必要的；完全均匀 batch 配额曾增加重复采样和过拟合风险，最终回到每 epoch 自然比例 shuffle + loss class weight 更合理。

## 9. 5-fold 实验状态

MCI 5-fold split 已正确按 subject/label 分层生成，train+validation 共 301 eligible subjects，原 test 保持独立。eye-only 与 eye+demographics 运行只完成了部分 fold：

- fold 0–2 两种分支的最佳 Val AUC 分别都是 0.8345、0.8090、0.8843；结果完全相同。
- fold 3 的 eye-only best 为 0.8079；demographics 分支没有完整 best summary。
- fold 4 未形成完整训练结果。

因此这组不能报告 5-fold mean，也不能据此判断 demographics 无效。前三 fold 完全相同还需要审计 demographic branch 是否真正影响 logits。本文不将它纳入最终选型。

## 10. 作废、未完成和不可比实验

以下结果不进入正式排序：

- 名称含 `failed_ddp_unused`、`failed_optimizer_group`、`aborted`、`stopped` 的运行。
- 使用错误 area-stat、旧 test 审计、旧 DDP rank 数据重复/覆盖、早停计数未清零或滚动窗口实现错误的运行。
- 旧 120-epoch layer sweep：只有 validation，没有最终 test，且数据/评估版本不同。
- 0.35 BERT mask：已停止，没有完整双下游。
- `Linear(16,4)` demographics：MCI 部分运行、PD5 未完成，不能与 raw16 完整比较。
- velocity 0.1/更大权重：短程未完成。
- V4 97755 新流水线曾排队后停止；只能引用早期 97755 完整结果。
- 当前 no-decay optimal search：仍在进行，报告生成时不具备 25/30/35/40K proxy 和三 seed 下游结果。

## 11. 当前推荐配置

### Tokenizer

- V5 packed view（由 V4 派生）、per-subject/per-eye median-MAD、clip 5、无 log1p。
- encoder 12×384、decoder 3×384、patch/stride 40、stim-isolated attention。
- 标准 FSQ `[9,7,5,5]`；FSQ 无 learned codebook，commitment loss 为 0。
- batch 128/GPU × 4，bf16，40K steps。
- LR `3e-4 → 3e-5`，warmup 2K 后 cosine。
- 使用第 4.4 节 loss，velocity=0，count feature 不参与 loss。
- checkpoint 最终必须由 25/30/35/40K 的 proxy + 双下游 validation 选择，当前暂用 40K。

### BERT

- 12×384、8 heads、FFN1152、dropout0。
- paired span 1–5、长度等概率、mask ratio 0.60；只 mask 双眼均满足 `nonmissing >= 0.85` 且非 padding 的时间 patch；stim 不 mask。
- factorized FSQ `[9,7,5,5]` 四头 CE；50K steps，LR `3e-4 → 3e-5`，warmup 2K 后 cosine。
- 30/40/50K checkpoint 仍需统一双下游筛选；当前暂用 50K。

### MCI / PD5

- strict four-task；K4 无放回；eval 使用全部有效 trials。
- `LayerNorm(CLS384) + demographics16`，共享 `400→128→output`，GELU，dropout0.3。
- trial logits 任务内平均，四任务 logits 均匀平均。
- embedding 冻结，解冻顶部 8 层，encoder/head LR 都为 `1e-5`，无 layer-wise decay。
- 100 epochs、warmup4、cosine、raw validation AUC early stop（min epoch27、patience20）。
- MCI subject positive weight、固定 threshold 0.5；PD5 inverse-frequency class weight、argmax。
- consistency 默认 0；MCI 可将 0.05 作为多 seed 次级候选。

## 12. 关键证据文件

- 数据版本完整流水线：
  - `outputs/eyevq/v3_fsq9755_recommended_tok40k_mask015_rawdemo16/downstream_test_summary.json`
  - `outputs/eyevq/v4_fsq9755_recommended_tok40k_mask015_rawdemo16/downstream_test_summary.json`
  - `outputs/eyevq/v31_fsq9755_recommended_tok40k_mask015_rawdemo16/downstream_test_summary.json`
- mask ratio：`outputs/eyevq/v4_ifsq9755_tok45k_mask_ablation/mask_ablation_comparison.json`
- LR 扫描：`outputs/eyevq/v4_ifsq9755_bs128_velocity0_lr_tuning/summary.json`
- FSQ 9755/97755：
  - `outputs/eyevq/fsq9755_demoproj128_mci_pd5_test_summary.json`
  - `outputs/eyevq/fsq97755_mci_pd5_test_summary.json`
- K2/K4：`outputs/eyevq/downstream_mci_k2_vs_k4_epoch100_comparison/comparison.json`
- 解冻层数：
  - `outputs/eyevq/mci_unfreeze6_vs8_h128_headlr1e5_cons005_e100_seed42/summary.json`
  - `outputs/eyevq/mci_unfreeze8_vs10_h128_headlr1e5_cons005_e100_seed42/summary.json`
- consistency：
  - `outputs/eyevq/mci_consistency_zero_vs005_h128_unfreeze8_headlr1e5_e100_seed42/summary.json`
  - `outputs/eyevq/mci_consistency_weight_h128_top6_headlr2e5_e100_seed42/summary.json`
- demographics/head：
  - `outputs/eyevq/mci_age_zscore_k4_trial_logit_mean_unfreeze8_headlr1e5_e100_seed42/summary_vs_raw_age.json`
  - `outputs/eyevq/mci_demographic_concat_multiview_k2x2_top6_headlr2e5_e100_seed42/summary.json`
- PD5 hidden/dropout：`outputs/eyevq/pd5_hidden_dropout_sweep_strict_shared_head_k4_top8_lr1e5_e100_seed42/summary.json`
- partial-task：
  - `outputs/eyevq/mci_age_zscore_k4_trial_logit_mean_partial_tasks_unweighted_unfreeze8_headlr1e5_e100_seed42/metrics_test.json`
  - `outputs/eyevq/pd5_logit_mean_partial_tasks_unweighted_unfreeze8_k4_zscore_e100_seed42/summary.json`

## 13. 最终解释边界

这批消融足以确定一套强而稳定的候选配置，但尚不足以宣称统计意义上的“全局最优”，原因是多数完整下游只跑了 seed42，部分实验同时改变了数据、tokenizer 和下游细节。最终可信结论需要当前 optimal-search 完成：

1. Tokenizer 25/30/35/40K 的独立 cache + BERT proxy；
2. BERT mask 0.15/0.25 的 30/40/50K checkpoint 双任务筛选；
3. top-{6,8,10} × LR-{5e-6,1e-5} 的 seed42 初筛；
4. 每个任务 validation 前两名补 seed43/44；
5. test 只做最终描述，不参与任何排序。
