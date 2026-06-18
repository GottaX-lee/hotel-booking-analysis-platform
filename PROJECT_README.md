# 酒店预订需求智能数据分析与可视化平台

## GitHub 仓库地址
https://github.com/GottaX-lee/hotel-booking-analysis-platform

## 使用方式

### 方式一：本地部署（推荐）
```bash
git clone https://github.com/GottaX-lee/hotel-booking-analysis-platform.git
cd hotel-booking-analysis-platform
pip install -r requirements.txt
python app.py
# 浏览器打开 http://localhost:5000
```

### 方式二：阿里云函数计算部署
1. 下载仓库 ZIP 包
2. FC 控制台创建函数，选"自定义运行时 Debian 10"
3. 上传 ZIP，启动命令 `./bootstrap`，监听端口 9000，内存 ≥ 1GB

测试数据：`batch_sample.csv` 含 20 条示例数据

---

## 一、项目概述

基于酒店预订真实数据集（阿里天池 Hotel Booking Demand，119,390 条记录），综合运用 XGBoost、PyTorch 神经网络、LSTM 时间序列、K-Means 聚类、SHAP 可解释性分析等技术，构建从数据分析到预测建模再到业务决策的完整闭环系统。

- **数据来源**：阿里天池平台，119,390 条，2 种酒店类型，177 个国家/地区
- **目标变量**：`is_canceled`（0=未取消，1=取消），取消率约 **37.5%**
- **课程关联**：神经网络与深度学习

---

## 二、核心功能模块（共 11 个）

| 模块 | 说明 |
|------|------|
| 首页 | 项目概览与技术架构图 |
| 数据看板 | 6 张 Plotly 交互式图表（KPI 卡片、取消率对比、月度趋势、特征分布、相关性热力图、模型对比） |
| 模型评估 | XGBoost 与神经网络双模型对比，混淆矩阵、特征重要性、Loss 曲线 |
| SHAP 可解释性分析 | 基于 TreeSHAP 的 Summary Plot，量化特征边际贡献 |
| 单样本预测 | 滑块调节 14+ 特征，实时计算取消概率 |
| 批量预测 | 上传 CSV，下载含 `predicted_canceled` 和 `cancel_probability` 的结果 |
| 风险预警 | 阈值可调（0.3~0.95），自动筛选高风险订单 |
| 场景模拟（What-If） | 修改参数对比基准场景，量化业务决策对取消率的影响 |
| 收益影响计算器 | 模拟降低取消率带来的收入提升、成本节约和 ROI |
| 客户细分画像 | K-Means 聚类识别 4 类客户群组 |
| 季节性预测 | LSTM 深度学习模型预测未来 12 个月订单量和取消率 |

---

## 三、模型性能

### XGBoost 分类模型（优化后）

| 指标 | 值 |
|------|:---:|
| **准确率（Accuracy）** | **0.8808** |
| **ROC-AUC** | **0.9546** |
| **精确率（Precision）** | 0.8644 |
| **召回率（Recall）** | 0.8089 |
| **F1-score** | 0.8357 |

### 分类报告

| 类别 | 精确率 | 召回率 | F1-score | 样本数 |
|:----:|:------:|:------:|:--------:|:------:|
| 未取消 | 0.89 | 0.92 | 0.91 | 14,684 |
| 取消 | 0.86 | 0.81 | 0.84 | 8,802 |

### 对比模型

| 模型 | 准确率 | AUC |
|------|:-----:|:---:|
| **XGBoost（优化）** | **0.8808** | **0.9546** |
| XGBoost（基线） | 0.8871 | 0.9571 |
| PyTorch 神经网络 | 0.7455 | 0.8428 |

---

## 四、与《神经网络与深度学习》课程关联

| 课程内容 | 项目对应 |
|----------|---------|
| RNN / LSTM 章节 | `src/lstm_forecast.py` — 2 层 LSTM 预测月度订单量 |
| MLP / 梯度下降 | `src/neural_model.py` — 3 层 MLP（256→128→64→1） |
| 模型可解释性 | `src/shap_analyzer.py` — TreeSHAP 算法 |
| 数据预处理 | Label Encoding、StandardScaler、异常值处理、数据泄露检测 |

---

## 五、Bug 修复记录（Git 历史可查）

| Bug | 修复 Commit | 问题 | 修复方案 |
|:---:|:----------:|------|---------|
| 1 | `0594e8f` | 数据泄露：`reservation_status` 列导致模型学到"未来信息" | 物理删除泄露列，准确率从 0.99 回归 0.89 |
| 2 | `e98692c` | ADR 异常值（≤0 或 >500）污染训练数据 | 剔除异常值 |
| 3 | 最新 | JSON 序列化失败：numpy.int64 和 NaN 无法被 json 模块解析 | 显式类型转换 + NaN 替换为 None |
| 4 | `5e18b5a` | SHAP 兼容性：xgboost base_score 格式不匹配 | 降级 xgboost 至 2.x |
| 5 | `13652da` | threadpoolctl Windows DLL 兼容性 | patch threadpool_limits + 使用 MiniBatchKMeans |

---

## 六、技术栈

- **语言**：Python 3.9
- **Web 框架**：Flask 3.0
- **前端**：Bootstrap 5 + Plotly.js
- **机器学习**：XGBoost 2.x, scikit-learn
- **深度学习**：PyTorch（MLP + LSTM）
- **可解释性**：SHAP 0.46
- **可视化**：Matplotlib, Seaborn, Plotly
- **部署**：阿里云函数计算 FC3

---

## 七、项目特色

1. **数据泄露危害量化实验** — 含泄露列准确率 0.99 vs 干净数据 0.89，差异约 10%
2. **静态 vs 交互式图表效能对比** — Matplotlib vs Plotly 的 3 个分析任务对比
3. **全链路 Git 版本控制** — 29 次提交，完整记录开发与 Bug 修复过程

---

*课程名称：神经网络与深度学习*
*提交日期：2026年6月*
