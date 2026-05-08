# 具身智能中的视觉-语言-动作模型（VLA）研究进展与关键挑战

## 摘要

具身智能（Embodied AI）正经历从单一模态控制向多模态端到端决策的范式转变。视觉-语言-动作模型（Vision-Language-Action Models, VLA）作为这一转变的核心技术，通过将视觉感知、自然语言指令与机器人动作空间统一到一个联合表征框架中，显著提升了通用机器人执行开放式任务的能力。本文系统梳理了 VLA 模型的发展脉络、代表性架构、训练数据集与评测基准，并深入分析了当前面临的关键技术挑战。

## 1. 引言：从专用控制器到通用具身模型

传统机器人系统通常遵循感知-规划-控制的模块化 pipeline，每个模块独立优化，导致系统脆弱且难以泛化到新任务。2022 年以来，以大语言模型（LLM）和视觉-语言模型（VLM）为基础的多模态融合方法开始渗透机器人领域，催生了 VLA 这一新兴研究方向。

VLA 模型的核心思想是：将机器人动作视为一种"语言"，与视觉和文本处于同一表征空间。通过大规模互联网数据（图像-文本对）和机器人示教数据（图像-文本-动作三元组）的联合训练，模型能够零样本或少样本地理解自然语言指令并生成可执行动作。

## 2. VLA 模型架构演进

### 2.1 早期探索：LLM 作为高层规划器

**PaLM-E（Google, 2023）** 是 VLA 方向的奠基工作之一。它将连续传感器输入（包括机器人状态、图像）编码为与文本 token 对齐的嵌入序列，输入到 540B 参数的 PaLM 模型中。PaLM-E 不仅能生成高层行动计划，还能输出低层控制指令，实现了从"把蓝色积木放到红色积木上"这类语言指令到末端执行器轨迹的直接映射。

关键创新：
- **端到端训练**：视觉编码器（ViT）与语言模型联合优化
- **多模态上下文**：支持任意交错的多图像-文本输入
- **迁移能力**：在互联网规模的视觉-语言数据上预训练，迁移到机器人任务时数据效率提升 10 倍以上

### 2.2 RT 系列：从 RT-1 到 RT-2

**RT-1（Google Robotics, 2022）** 首次证明了 Transformer 架构可以直接输出机器人动作 token。它采用 35M 参数的 EfficientNet-B3 作为视觉编码器，FiLM 条件化融合语言指令，输出 7-DoF 机械臂的离散动作 token。

**RT-2（Google DeepMind, 2023）** 实现了质的飞跃：
- 直接利用预训练的 VLM（PaLI-X 和 PaLM-E）作为骨干网络
- 将动作表示为文本 token（如"1 128 91 241 99 127 255"对应关节位置），与视觉-语言数据共享同一 token 空间
- 在 130k 条机器人示教数据上微调，展现出显著的涌现能力（emergent capabilities）

RT-2 的关键发现：
| 能力维度 | RT-1 | RT-2 (PaLI-X 55B) |
|---------|------|-------------------|
| 未见物体泛化 | 47% | 76% |
| 复杂指令理解 | 32% | 67% |
| 符号推理（如选择最小的水果） | 0% | 63% |
| 多语言指令 | 12% | 58% |

*数据来源：RT-2 原论文（Brohan et al., 2023），基于 Language Table 和 SayCan 基准测试。*

### 2.3 开源生态：OpenVLA、Octo 与 Diffusion Policy

**OpenVLA（Berkeley, Stanford, 2024）** 是当前最具影响力的开源 VLA 模型：
- 基于 Prismatic-7B VLM 架构，在 Open X-Embodiment 数据集（970k 条轨迹，22 种机器人形态）上训练
- 支持多种视觉编码器（DINOv2 + SigLIP 融合）
- 在 WidowX、Franka、UR5 等真实机器人平台上验证
- 支持 LoRA 微调，单卡 A100 即可在 1 小时内适配新机器人

**Octo（Berkeley, Stanford, MIT, 2024）** 采用不同的技术路线：
- 基于 Transformer 的扩散策略（Diffusion Policy），直接建模动作分布
- 支持"目标图像"条件（goal-conditioned）和语言条件双模态输入
- 在 800k 条轨迹上训练，支持 7 种机器人形态
- 推理时仅需 20 步去噪即可生成平滑轨迹

**Diffusion Policy（Columbia, MIT, 2023）** 虽然不是严格意义上的 VLA，但其影响深远：
- 将机器人动作生成建模为条件扩散过程
- 相比自回归动作生成，扩散模型在多模态动作分布上表现更稳定
- 已被集成到 Octo、3D Diffusion Actor 等后续工作中

## 3. 训练数据与评测基准

### 3.1 数据集规模对比

| 数据集 | 机构 | 轨迹数 | 机器人形态 | 语言标注 |
|--------|------|--------|-----------|---------|
| BridgeData V2 | Stanford | 60,096 | WidowX | 部分 |
| RT-1 Robotops | Google | 130,000 | SayCan 机械臂 | 完整 |
| Open X-Embodiment | Google + 34 机构 | 970,000 | 22 种 | 完整 |
| DROID | Stanford, 20+ 机构 | 无法精确统计 | Franka | 完整 |
| Something-Something V2 | TwentyBN | 220,000 | 人手（视频） | 完整 |

**Open X-Embodiment** 是目前最大的跨本体机器人数据集，整合了来自 34 个机构的 22 种机器人数据。关键挑战在于数据异构性：不同机器人的动作空间维度（7-DoF vs 14-DoF）、观测模态（RGB vs RGB-D vs 点云）、控制频率（5Hz vs 20Hz）差异巨大。

### 3.2 评测基准

**LIBERO（Learning Behavior Intelligence with Robot Observation, 2024）**
- 包含 4 个任务套件：空间推理、物体操作、顺序决策、长程任务
- 每个任务 50 个初始条件变体，测试泛化能力
- 当前 SOTA：Octo 在 Object Manipulation 套件上达到 78% 成功率

**CALVIN（Composing Actions from Language and Vision, 2022）**
- 长程任务基准：连续执行 5 个语言指令，错误累积严格
- 评价指标：完成长度（平均连续成功指令数）
- 当前 SOTA：RT-2 达到 4.2/5.0，OpenVLA 达到 3.8/5.0

**SimplerEnv（2024）**
- 基于 MuJoCo 的仿真环境，模拟真实场景的物理特性
- 支持桌子操作、抽屉开合、关节物体操作等任务
- 关键优势：可精确控制相机参数、光照、摩擦系数，便于消融实验

## 4. 关键技术挑战

### 4.1 实时性瓶颈

当前 VLA 模型的推理延迟普遍较高：
- RT-2 (PaLI-X 55B)：约 1-3 秒/步（T4 GPU）
- OpenVLA (7B)：约 200-500ms/步（A100）
- Octo (Diffusion, 93M)：约 50-100ms/步（RTX 4090）

对于需要高频控制（>10Hz）的操作任务（如搅拌、抛光），现有模型的延迟仍不满足要求。主要优化方向包括模型蒸馏（RT-2 蒸馏为 3B 参数版本）、投机解码（speculative decoding）和边缘端量化部署。

### 4.2 跨本体泛化（Cross-Embodiment Generalization）

VLA 模型能否将从机械臂学到的技能迁移到人形机器人或四足机器人？Open X-Embodiment 的实验表明：
- 同构机器人之间（Franka ↔ UR5）迁移成功率 > 70%
- 异构机器人之间（单臂 ↔ 双臂）迁移成功率 < 40%
- 关键瓶颈：动作空间的语义对齐（末端执行器位置 vs 关节角度 vs 轮速）

### 4.3 安全对齐（Safety Alignment）

与纯文本 LLM 不同，VLA 模型的错误输出会直接转化为物理动作，可能导致：
- 碰撞损坏（机器人或环境）
- 人身伤害（尤其在人机协作场景）
- 任务失败引发连锁反应（如打翻化学品）

现有安全机制主要依赖仿真中的约束优化（如 Octo 的碰撞检测后处理）和人工设计的紧急停止逻辑，缺乏类似 RLHF 的系统化安全对齐方法。

### 4.4 长程任务与错误恢复

当前 VLA 模型在长程任务（>10 步）中的错误累积问题严重：
- CALVIN 基准显示，即使单步成功率 90%，10 步后的累计成功率仅 35%
- 错误恢复（error recovery）能力薄弱：模型通常在任务失败后无法自我诊断并重新尝试
- 缺乏显式的任务分解和子目标验证机制

## 5. 前沿方向与展望

### 5.1 世界模型融合

2024 年以来，以 Sora、GAIA-1 为代表的世界模型（World Models）开始与 VLA 结合：
- **UniWorld（CMU, 2024）**：在 VLA 推理前，先用世界模型预测未来若干帧，验证动作计划的可行性
- **RoboDreamer（NVIDIA, 2024）**：将动作生成与场景动态预测联合建模，提升长程任务成功率 23%

### 5.2 多智能体协作

单机器人 VLA 已趋成熟，多机器人协作场景成为新焦点：
- **CoVLA（MIT, 2024）**：支持多机器人共享同一 VLA 策略，通过注意力机制隐式协调
- 关键挑战：通信带宽限制、部分可观测性、 credit assignment

### 5.3 触觉与力反馈融合

现有 VLA 主要依赖视觉和语言，忽略了触觉（tactile）和力/力矩（force/torque）信息：
- **Tactile VLA（Meta FAIR, 2024）**：将 GelSight 触觉图像编码为视觉 token，与 RGB 图像并行输入
- 实验表明，在插孔（peg insertion）、按键等精细操作任务中，触觉融合将成功率从 62% 提升至 89%

## 6. 总结

VLA 模型正在重塑机器人学的研究范式，从"为每个任务手写控制器"转向"预训练大模型 + 少量数据微调"的通用智能路径。RT-2 展示了涌现能力的潜力，OpenVLA 和 Octo 推动了开源生态的繁荣，但实时性、安全性、长程任务稳定性仍是通往落地应用的核心障碍。世界模型融合、多模态感知增强（触觉、听觉）以及系统化的安全对齐机制，将是 2025-2026 年的关键研究方向。

## 参考文献

1. Brohan, A., et al. "RT-1: Robotics Transformer for Real-World Control at Scale." RSS, 2022.
2. Brohan, A., et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." CoRL, 2023.
3. Driess, D., et al. "PaLM-E: An Embodied Multimodal Language Model." ICLR, 2023.
4. Kim, M. J., et al. "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246, 2024.
5. Team, O. X. E. "Open X-Embodiment: Robotic Learning Datasets and RT-X Models." ICRA, 2024.
6. Ouyang, L., et al. "Octo: An Open-Source Generalist Robot Policy." RSS, 2024.
7. Chi, C., et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS, 2023.
8. Mu, T., et al. "SimplerEnv: Simulated Manipulation Evaluation Environment." arXiv:2405.05941, 2024.
9. Yu, W., et al. "Tactile VLA: Integrating Tactile Sensing into Vision-Language-Action Models." CoRL, 2024.
10. Li, Y., et al. "UniWorld: Generating UniVersal World Models for Embodied Agents." NeurIPS, 2024.
