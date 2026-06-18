# Jane Street Market Prediction — 项目决策记录

> 项目路径: `CodeBuddy/Jane_Street_Market/`
> 比赛: [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
> 最后更新: 2026-06-18

## 零、远程 GPU 管理工具调研 (2026-06-18)

### OpenClaw
- **GitHub**: openclaw/openclaw
- **本质**: 个人 AI 助手网关，把 LLM 接入 WhatsApp/微信/Slack/Discord 等多渠道
- **MIT 协议**，需要 Node 24
- **与 GPU 训练无关** ✗

### Hermes Agent
- **GitHub**: NousResearch/hermes-agent
- **本质**: 自我进化的 AI Agent 框架
- **有用功能**: SSH backend（连远程机）、cron 任务调度、web dashboard、子 Agent 委派
- **潜在用途**: 常驻 GPU 服务器 → web UI 监控训练进度 → cron 自动串下一步
- **当前决策**: 暂不使用。单台 4090 不需要额外服务层。等 6×2080 Ti 到了再评估。

### 当前方案
```
Mac → sshpass → GPU 服务器 → nohup 后台跑 → 定时检查 → 跑完自动下一步
```
简单够用，不引入额外依赖。

---

## 一、数据概况

| 维度 | 数值 |
|------|------|
| 训练总行数 | 47,127,338 |
| 日期范围 | date_id 0 ~ 1698（1699 天） |
| 日内时间点 | 968 个 time_id |
| 交易标的 | 39 个 symbol_id（0~38） |
| 特征列 | 79 个 feature_00~78 + 9 个 responder_0~8 + 9 个 lag |
| 目标列 | `responder_6`（连续值，clip 在 [-5, 5]，均值≈0, std≈0.89） |
| 评估指标 | Weighted R² |
| 原始大小 | ~12 GB（10 个 Hive 分区） |

### 数据来源

比赛官方数据无法通过 Kaggle API 下载（比赛已截止），使用公开 Dataset:
`mohamedsameh0410/jane-street-dataset`，通过 `kagglehub` 下载。

---

## 二、缺失值分析

### 缺失值覆盖

- 79 个 feature 中，最多 47 个有缺失值（取决于数据时间段）
- **每行至少有一列 NaN**，无法删行（`drop_nulls()` 后剩 0 行）
- 4 个元数据列 + 9 个 responder 列无缺失

### 三类缺失模式

#### 类型 1：时间引入型（9 个特征）
```
feature_00~04, feature_21, feature_26, feature_27, feature_31
```
- 早期 date_id (0~150) 100% NaN，中后期完全正常
- 分布：接近正态，mean≈0, std≈1
- **原因判断**：后期才加入的数据源

#### 类型 2：周期性缺失（4 个特征）
```
feature_39, feature_42, feature_50, feature_53
```
- 每天约 7% 的 symbol 缺失，所有 symbol 同步
- 分布：接近正态
- **原因判断**：特定日期的数据源故障

#### 类型 3：偶发缺失（~30 个特征）
```
feature_15, feature_62~66, feature_73~78 等
```
- 缺失率 <3%，分散在个别 date × symbol 组合
- **原因判断**：偶发数据质量问题

---

## 三、填充策略

### 核心发现：特征分为"慢变量"和"快变量"

| 类型 | 日间自相关 | 日变化/总体σ | 代表特征 |
|------|-----------|-------------|---------|
| 慢变量 | +0.6 ~ +0.9 | 0.3 ~ 0.7 | feature_00, 02, 03, 15, 39, 42, 50, 53 |
| 快变量 | -0.2 ~ +0.1 | 1.0 ~ 1.2 | feature_01, 04, 05, 06, 07, 08 |

### 分层填充

| 层 | 特征 | 策略 | 理由 |
|----|------|------|------|
| 慢变量 | feature_00, 02, 03, 15, 39, 42, 50, 53 | **forward fill**（按 symbol_id 分组） | 日间自相关高，昨天值误差小 |
| 快变量 | feature_01, 04 | **填 0** | 日间跳跃大，forward fill 无效，0 ≈ 均值 |
| 其余 | 所有其他 feature | **forward fill → 0** | 先尝试 ffill，剩余填 0 |

### 实施

```python
# 按 symbol 分组 → 按 date_id 排序 → forward fill → 剩余填 0
df = df.sort(["symbol_id", "date_id", "time_id"])
df = df.with_columns([
    pl.col(STABLE_FFILL).fill_null(strategy="forward").over("symbol_id")
])
df = df.with_columns([pl.col(UNSTABLE_FILL_ZERO).fill_null(0)])
# 其余: ffill + 0
df = df.fill_null(0)
```

---

## 四、特征清单

| 类别 | 数量 | 说明 |
|------|------|------|
| 匿名连续特征 | 76 | feature_00~78，除去 09/10/11 是分类特征 |
| 匿名分类特征 | 3 | feature_09 (17类), feature_10 (9类), feature_11 (21类) |
| 历史响应值 | 8 | responder_0~5, 7~8（不含 target responder_6） |
| Lag 特征 | 9 | responder_0_lag_1 ~ responder_8_lag_1 |
| **建模特征总数** | **96** | |

---

## 五、信噪比

- 单一特征与 target 的最强相关性: |r| = 0.064（feature_06）
- 大多数特征 |r| < 0.03
- Target 日间自相关: 0.18（近乎白噪声）
- **结论**: 信号极弱且分散在大量特征中，需用模型集成捕捉微弱模式

---

## 六、后续待定

- [x] 特征标准化方案 → StandardScaler（均值0 方差1）
- [x] 训练/验证集划分策略 → 按 date_id 时间切分（避免数据泄露）
- [x] 模型选择 → Ridge(基线) → CatBoost → MLP → 集成
- [ ] 集成权重调优方案
- [ ] 推理 pipeline
- [x] Ridge 基线 — R² = 0.0040 (本地 CPU, 7.4min)
- [ ] CatBoost 基线 — 代码就绪, 待 GPU 运行
- [ ] MLP 模型
- [ ] 三层集成

---

## 九、当前状态 (2026-06-06)

### 已完成
- 数据下载 + 预处理 (47M rows × 101 cols, 无 NaN)
- EDA: 信噪比分析、缺失值模式、特征关联
- TDA 特征分析: 五大族群 + 17 孤立特征
- Ridge 基线: R²=0.0040 (147 特征)
- **GPU 训练代码就绪** (CatBoost + XGBoost + MLP + 集成)

### 待完成
- [ ] GPU 机器上运行 `train_all.py`
- [ ] 推理 pipeline

### 阻塞项
- ~~无 GPU~~ → 借到 6 × 2080 Ti, 环境待部署

---

## 十、GPU 训练代码架构 (2026-06-06)

### 决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 树模型采样 | A: 均匀 1/6 (~6.5M rows) | 覆盖全面，无 regime shift |
| MLP 架构 | [512, 512, 256] | 更深的网络，但有监控降级 |
| MLP 特征 | v1 (88维) + v2 (TDA 42维) 都做 | 对比验证 |
| XGBoost | 有时间就跑 | 作为集成成员，不是必须 |
| 最终输出 | 模型文件 | 不生成 submission |

### 文件结构

```
Jane_Street_Market/
├── train_all.py              # 主入口，一键顺序训练
├── check_env.py              # GPU 环境预检查
├── setup_gpu.sh              # 自动检测 CUDA + 安装依赖
├── SETUP_GPU.md              # GPU 机器部署指南
├── requirements_gpu.txt      # Python 依赖
├── src/
│   ├── metrics.py            # 统一 weighted R²
│   ├── data_utils.py         # 数据加载 + TDA 聚类 + 特征工程
│   ├── train_catboost.py     # CatBoost GPU (6 GPU parallel)
│   ├── train_mlp.py          # MLP (v1 88dim + v2 42dim)
│   ├── train_xgboost.py      # XGBoost GPU (multi GPU)
│   ├── ensemble.py           # 集成权重搜索
│   ├── preprocess.py         # [已有] 数据预处理
│   └── ridge_baseline.py     # [已有] Ridge 基线
└── models/                   # 输出: .cbm, .pth, .json, .npy
```

### 容错设计

- 每个模型独立 try/catch，失败跳过，成功模型不丢失
- 每个模型训练完立即保存 checkpoint
- CatBoost → MLP v1 → MLP v2 → XGBoost → 集成（顺序执行）
- MLP 内置监控：loss不降→降LR；VRAM>10GB→降batch；负R²→降级架构

### 多 GPU 策略 (6 × 2080 Ti)

| 模型 | GPU 使用 | 说明 |
|------|---------|------|
| CatBoost | 0-5 (全部) | devices='0-5', 并行 boosting |
| XGBoost | 0-5 (全部) | device='cuda', 自动多 GPU |
| MLP v1 | 单 GPU 0 | 模型小(442K 参数), 单 GPU 够 |
| MLP v2 | 单 GPU 0 | 同上 |

---

## 八、模型策略（基于 Zhao & Yu 2006 论文结论）

### 为什么 Ridge 而非 LASSO

论文实验验证了 Irrepresentable Condition：
- ρ > 0.5 时 IC 违反，Lasso 从相关组中**任意选一个**变量，其余归零
- 我们的 TDA 发现 cluster_1 有 46 个特征 |corr| > 0.5，cluster_0/1 内 |corr| > 0.65
- **Lasso 会在 12 个相关特征中随机选一个，每日可能不同 → 预测方差大**
- Ridge 将权重分摊到组内所有特征 → 更稳健
- Adaptive Lasso 值得探索：Ridge 作为初始权重 + Lasso 稀疏化

### 为什么 AdamW 而非 SGD

| | SGD | AdamW |
|------|-----|-------|
| 收敛速度 | 慢，依赖学习率 | 快，自适应 |
| 特征尺度敏感 | 需精细 scaling | 自适应 |
| Weight decay | 手动衰减 | 解耦式，与 λ 独立调 |

论文实验 5 确认 λ 是关键超参，AdamW 的解耦 weight decay 允许独立调学习率和正则化强度。

### 三树模型集成分析

| 组合 | 预期预测相关 ρ̄ | 方差缩减 |
|------|:---:|------|
| LightGBM + XGBoost + CatBoost | ≈0.87 | **仅 -9%** |
| Ridge + CatBoost | ≈0.60 | **-47%** |
| Ridge + CatBoost + MLP | ≈0.50 | **-67%** |

**结论**: 三树互相关联太强，集成效果有限。最终方案:

```
CatBoost  → 主力树模型（ordered boosting + 原生分类特征）
XGBoost   → 对照验证
LightGBM  → 快速特征工程迭代（训练快）

最终集成: Ridge + CatBoost + MLP  跨范式集成
           (线性)   (树)     (NN)
```

---

## 七、TDA 特征关联分析

### 方法

- 将每个 feature 作为点，其在 968 个 time_id 上的取值作为坐标
- 距离：`1 - |correlation|`
- 随机采样 200 个 (date_id, symbol_id) 组合，验证结构稳定性
- 单链接聚类

### 五大特征族群（r=0.35, |corr| > 0.65）

| 族群 | 成员数 | 代表特征 | 平均 |corr| |
|------|--------|---------|----------|
| cluster_0 | 12 | feature_01, 08, 18, 37, 38, 46, 57, 65 | 0.62 |
| cluster_1 | 12 | **feature_05**, 07, 39, 42, 45, 47, 49, 50 | 0.65 |
| cluster_2 | 3 | feature_12, 67, 70 | 0.81 |
| cluster_3 | 3 | feature_13, 68, 71 | 0.79 |
| cluster_4 | 3 | feature_14, 69, 72 | 0.81 |

### 17 个孤立特征（最具独立信息量）

```
feature_00, 04, 15, 16, 17, 19, 32, 34, 35, 36, 40, 43, 51, 62, 63, 64, 66
```
这些特征与任何其他特征的 |corr| < 0.5，无法被其他特征替代。

### H1 环分析

- 环仅在族群内部出现（如 05↔07↔60↔05），无跨族群循环
- 五个族群之间的拓扑关系是**树状**的
- H1 未发现跨族群的周期性依赖

### 稳定性

200 次随机采样的 H0 聚类结构 **100% 一致**。特征关联是数据源的**结构属性**，不随日期/symbol 改变。

### 建模建议

```
特征压缩方案:
  5 个族群代表（PCA 第一主成分） → 5 维
  + 17 个孤立特征                 → 17 维
  + 3 个分类特征 (09, 10, 11)     → 3 维
  + 8 个 responder (非target)     → 8 维
  + 9 个 lag                       → 9 维
  = 42 维精简特征集
```
