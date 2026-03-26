# Kepler 恒星自转周期测量项目

本仓库是毕业论文配套代码与实验记录，围绕 Kepler 恒星光变数据开展自转周期 `Prot` 测量与对比研究。项目包含：

- 数据准备与光变下载
- 传统周期测量基线：`LS / ACF / GPS / QP-GP`
- 端到端机器学习探索：`LightPred / CNN / LSTM`
- 混合模型：`QP-GP` 残差校正与 `CNN` 特征融合
- Notebook 结果分析与论文配图

## 1. 环境依赖

建议使用 Python `3.10+`。

基础依赖：

```bash
pip install -r requirements.txt
```

如果需要重新下载 catalog 或 light curve，还需要：

```bash
pip install lightkurve astroquery scipy
```

如果需要运行 `QP-GP`，还需要保证相关 GP 依赖可用，例如 `celerite2`。

## 2. 项目结构

```text
.
├─ prepare_kepler_mcquillan.py
├─ src/
│  ├─ analysis/
│  │  ├─ baseline_plots.py
│  │  └─ npz_overview.py
│  ├─ gp/
│  │  └─ qpgp.py
│  ├─ io/
│  │  └─ kepler_loader.py
│  ├─ traditional/
│  │  ├─ lomb_scargle.py
│  │  ├─ acf.py
│  │  ├─ wavelet_gps.py
│  │  └─ run_baseline_periods.py
│  └─ lightpred/
│     ├─ data.py
│     ├─ model.py
│     ├─ train_lightpred.py
│     ├─ train_lightpred_default.py
│     ├─ train_cnn_default.py
│     ├─ train_lstm_default.py
│     ├─ evaluate_lightpred.py
│     ├─ train_hybrid_models.py
│     ├─ generate_shared_val_notebook.py
│     └─ generate_hybrid_comparison_notebook.py
├─ data/kepler/
│  ├─ mcquillan2014_catalog.csv
│  ├─ mcquillan2014_sample.csv
│  ├─ baseline_periods_ls_acf.csv
│  ├─ lc_overview.csv
│  ├─ lightcurves/
│  └─ models/
├─ baseline_periods_runner.ipynb
├─ kepler_dataset_overview.ipynb
├─ baseline_result_analysis.ipynb
├─ lightpred_result_analysis.ipynb
├─ ml_model_comparison.ipynb
├─ shared_val_model_comparison.ipynb
├─ hybrid_model_comparison.ipynb
└─ APPENDIX_PROJECT.md
```

## 3. 数据与输出说明

核心数据文件：

- `data/kepler/mcquillan2014_catalog.csv`
  - McQuillan+2014 目录，至少包含 `kic` 与 `prot`
- `data/kepler/lightcurves/kic#########.npz`
  - 单星光变，包含 `time / flux / flux_err`
- `data/kepler/baseline_periods_ls_acf.csv`
  - baseline 输出，包含 `prot_ls / prot_acf / prot_gps / prot_qpgp`
- `data/kepler/lc_overview.csv`
  - 光变统计量，如 `n_points / time_span_days / flux_std / teff`
- `data/kepler/models/`
  - 机器学习与 hybrid 的模型、预测结果和汇总表

## 4. 快速开始

### 4.1 下载 catalog 与光变

```bash
python prepare_kepler_mcquillan.py \
  --n-sample 50 \
  --min-prot 0.5 \
  --max-prot 40.0 \
  --max-download 50
```

说明：

- 默认脚本会尝试使用本地代理；如不需要，可加 `--noproxy`
- 如果只是复现当前论文结果，通常不需要重新下载

### 4.2 运行传统 baseline

```bash
python -m src.traditional.run_baseline_periods --max-targets 50
```

可选参数：

- `--skip-gps`
- `--skip-qpgp`
- `--save-every`

输出文件：

- `data/kepler/baseline_periods_ls_acf.csv`

## 5. LightPred / CNN / LSTM

### 5.1 LightPred

`LightPred` 是一个双分支模型：

- `LSTM` 分支提取局部时序特征
- `Transformer` 分支提取长程依赖
- 最终回归 `log(Prot)` 与 `log(sigma)`

默认训练入口：

```bash
python -m src.lightpred.train_lightpred_default
```

通用训练入口：

```bash
python -m src.lightpred.train_lightpred \
  --model-type lightpred \
  --epochs 150 \
  --batch-size 8 \
  --seq-len 4096
```

### 5.2 CNN

默认训练入口：

```bash
python -m src.lightpred.train_cnn_default
```

### 5.3 LSTM

默认训练入口：

```bash
python -m src.lightpred.train_lstm_default
```

### 5.4 模型评估

```bash
python -m src.lightpred.evaluate_lightpred \
  --checkpoint data/kepler/models/cnn/cnn_best.pt \
  --out-dir data/kepler/models/cnn
```

说明：

- `evaluate_lightpred.py` 现在支持 `LightPred / CNN / LSTM`
- 会自动按 checkpoint 中保存的参数重建模型与数据划分

常见输出：

- `*_predictions.csv`
- `*_eval_summary.csv`
- `*_baseline_compare_val.csv`

## 6. Hybrid 模型

Hybrid 模型在不增加新数据的前提下，融合 baseline、机器学习预测与光变统计特征。

训练入口：

```bash
python -m src.lightpred.train_hybrid_models
```

当前实现了两条路线：

### 6.1 Hybrid-QPResidual

思路：

- 以 `QP-GP` 为主预测
- 学习残差 `P_label - P_qpgp`
- 最终输出 `P_qpgp + residual_correction`

主要输入包括：

- `prot_qpgp / prot_ls / prot_gps / prot_acf`
- `qpgp_logL / qpgp_success / q_gps`
- `cnn_pred / lightpred_pred / lstm_pred`
- 方法间差值特征，如 `ls_qpgp_gap / cnn_qpgp_gap`
- 光变统计量与 `teff`

### 6.2 Hybrid-CNNFusion

思路：

- 以 `CNN` 预测为主要机器学习特征
- 联合 baseline 结果与统计特征
- 直接回归最终 `Prot`

当前结果文件在：

- `data/kepler/models/hybrid_qpgp_residual/`
- `data/kepler/models/hybrid_cnn_fusion/`

## 7. Notebook 说明

### 数据与 baseline

- `kepler_dataset_overview.ipynb`
  - 数据集规模、分布与基础统计
- `baseline_periods_runner.ipynb`
  - 批量运行 baseline
- `baseline_result_analysis.ipynb`
  - baseline 误差分布、散点图、指标汇总

### 机器学习

- `lightpred_training_visualization.ipynb`
  - LightPred 训练曲线
- `lightpred_result_analysis.ipynb`
  - LightPred 预测表现
- `ml_model_comparison.ipynb`
  - `LightPred / CNN / LSTM` 对比

### 同一验证集对比

- `shared_val_model_comparison.ipynb`
  - 同一批验证样本上的 `baseline + ML` 对比
- `hybrid_model_comparison.ipynb`
  - `Hybrid-QPResidual / Hybrid-CNNFusion` 与其他方法统一比较

## 8. 当前实验结论概览

在当前固定验证集上：

- `QP-GP` 是最强的单一传统方法
- `CNN` 是当前最有效的纯机器学习模型
- `LightPred / LSTM` 在小样本下表现一般
- `Hybrid-QPResidual` 与 `Hybrid-CNNFusion` 明显优于单独的 `QP-GP` 与 `CNN`

如果你是为了论文复现，建议按这个顺序查看：

1. `kepler_dataset_overview.ipynb`
2. `baseline_result_analysis.ipynb`
3. `shared_val_model_comparison.ipynb`
4. `hybrid_model_comparison.ipynb`

## 9. 注意事项

- 本仓库中部分结果文件已经生成，不必每次从头训练
- `ACF` 在共享验证集上有效样本较少，解释时要单独说明 `N`
- `Hybrid` 结果目前基于固定 train/val 划分，不等同于完整交叉验证结论
- 如果 `README.md` 在其他编辑器里显示异常，请确认文件编码为 `UTF-8` 无 BOM
