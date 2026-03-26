# 项目说明（论文附录）

## 1. 项目目的
本项目用于构建与评估 Kepler 恒星自转周期测量的实验管线，包含：
- 数据获取与预处理；
- 传统基线方法（LS/ACF/GPS/QP-GP）；
- 深度模型 LightPred（LSTM + Transformer）；
- 统一的结果输出与可视化。

## 2. 数据来源
- **McQuillan+2014** Kepler 自转周期目录（VizieR: J/ApJS/211/24）。
- Kepler PDC-SAP 光变（MAST/Lightkurve）。

## 3. 数据处理流程
1) 从 VizieR 下载 McQuillan+2014 目录并保存为 CSV。  
2) 按周期范围筛选并随机抽样目标。  
3) 通过 Lightkurve 下载对应的 Kepler 光变，保存为 `npz`（`time/flux/flux_err`）。  
4) 对光变进行轻量级预处理：去 NaN、粗略去趋势与归一化。  

## 4. 方法与模型
### 4.1 传统基线
位于 `src/traditional/`：
- Lomb–Scargle（LS）
- 自相关函数（ACF）
- Wavelet + GPS
- QP-GP（celerite2 RotationTerm）

基线输出文件：  
`data/kepler/baseline_periods_ls_acf.csv`

### 4.2 LightPred（论文主体模型）
位于 `src/lightpred/`：
双分支结构：  
- **LSTM** 建模局部时序模式  
- **Transformer** 捕获长程依赖  
融合后回归 `log(Prot)` 和 `log(sigma)`（异方差高斯回归）。

训练脚本：  
- `src/lightpred/train_lightpred.py`（可配置参数）  
- `src/lightpred/train_lightpred_default.py`（固定参数，直接可跑）

## 5. 训练与评估
训练目标：
- 回归 `log(Prot)`，使用高斯 NLL 损失；
- 评估指标：**Period MAE (days)** 与 **NLL**。

训练日志：
- `data/kepler/models/lightpred_train_log.csv`

可视化：
- `lightpred_training_visualization.ipynb`

## 6. 复现实验步骤
1) 安装依赖：
```bash
pip install -r requirements.txt
pip install lightkurve astroquery scipy
```
2) 下载数据：
```bash
python prepare_kepler_mcquillan.py --n-sample 50 --max-download 50
```
3) 运行传统基线：
```bash
python -m src.traditional.run_baseline_periods --max-targets 50
```
4) 训练 LightPred（固定参数）：
```bash
python -m src.lightpred.train_lightpred_default
```
5) 打开 `lightpred_training_visualization.ipynb` 查看训练曲线。

## 7. 目录结构
```
.
├─ prepare_kepler_mcquillan.py
├─ src/
│  ├─ io/kepler_loader.py
│  ├─ traditional/
│  ├─ gp/
│  └─ lightpred/
├─ data/kepler/
│  ├─ mcquillan2014_catalog.csv
│  ├─ mcquillan2014_sample.csv
│  ├─ lightcurves/
│  ├─ baseline_periods_ls_acf.csv
│  └─ models/
└─ lightpred_training_visualization.ipynb
```

## 8. 运行环境
- Python 3.10+
- 主要依赖：numpy、pandas、scikit-learn、astropy、torch
- 额外依赖：lightkurve、astroquery、scipy

## 9. 已知限制与改进方向
- 目前样本量相对较小，深度模型易过拟合。  
  可通过多片段训练、数据增强与更严格正则化缓解。  
- 训练采用单一随机划分，后续可引入交叉验证或星团/星震基准集进行更稳健评估。  

---
如需将本项目作为论文附录引用，可直接引用本说明，并结合实验章节提供的结果与图表。
