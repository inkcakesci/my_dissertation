# Kepler 自转周期样本与基线

本仓库是毕业论文的配套实验代码。基于 McQuillan+2014 公布的 Kepler 恒星自转周期目录，提供数据获取脚本、传统（LS/ACF）基线实现，以及示例数据与结果，便于后续自转周期研究或模型对比。

## 功能概览
- 从 VizieR 自动下载 McQuillan+2014 目录并保存为 CSV。
- 按周期范围筛选、随机抽样目标，生成样本列表。
- 通过 lightkurve 从 MAST 下载对应的 PDC-SAP 光变，保存为 `npz`（time/flux/flux_err）。
- 运行 Lomb–Scargle 与自相关函数（ACF）两种传统算法，得到基线周期估计 CSV。

## 环境依赖
建议使用 Python 3.10+。核心依赖（未写入 `requirements.txt`，请手动安装）：
```bash
pip install lightkurve astroquery pandas numpy matplotlib astropy scipy
```

## 目录结构
- 项目根目录概览：
```
.
├─ prepare_kepler_mcquillan.py    # 主数据准备脚本：下载目录、筛选样本、抓取光变
├─ src/
│  ├─ io/kepler_loader.py         # 读取并预处理单颗 Kepler 光变 npz
│  └─ traditional/
│     ├─ lomb_scargle.py          # Lomb–Scargle 周期估计
│     ├─ acf.py                   # 自相关函数（ACF）周期估计
│     └─ run_baseline_periods.py  # 批量跑 LS/ACF 基线，输出 CSV
├─ data/kepler/
│  ├─ mcquillan2014_catalog.csv   # 已下载的完整目录
│  ├─ mcquillan2014_sample.csv    # 筛选后的随机样本（含标注周期）
│  ├─ baseline_periods_ls_acf.csv # 基线输出示例
│  └─ lightcurves/                # 下载的光变 npz：time/flux/flux_err
├─ part1.ipynb, plot.ipynb        # 探索/可视化笔记本
└─ Reference/                     # 论文相关参考资料（如有）
```

## 数据流程与目标
- 论文实验数据链路：VizieR 下载目录 → 周期筛选与随机采样 → MAST 下载光变 → 传统基线（LS/ACF）估计 → 输出对比用 CSV。
- 提供的 CSV 和 npz 可直接复现实验，或作为后续模型的输入/标签。

## 快速开始
1) **下载目录与光变**（默认尝试本地代理 `127.0.0.1:7890`；若不需要可加 `--noproxy`）：
```bash
python prepare_kepler_mcquillan.py \
  --n-sample 50 \
  --min-prot 0.5 \
  --max-prot 40.0 \
  --max-download 50   # 可选：仅下载前 N 颗做快速测试
```
生成文件：
- 目录：`data/kepler/mcquillan2014_catalog.csv`
- 样本：`data/kepler/mcquillan2014_sample.csv`
- 光变：`data/kepler/lightcurves/kic#########.npz`

2) **运行传统基线**（在已下载光变的基础上）：
```bash
python -m src.traditional.run_baseline_periods --max-targets 50
```
输出：`data/kepler/baseline_periods_ls_acf.csv`，包含 `kic, prot_label, prot_ls, prot_acf`。

## 额外说明
- 代理：脚本默认设置 `HTTP(S)_PROXY=127.0.0.1:7890`，若已配置环境变量或无需代理，可添加 `--noproxy` 避免覆盖。
- 数据质量：下载的光变会做 NaN/Inf 掩蔽与粗略去趋势/归一化；若要应用更复杂的去系统atics，请在加载后自行处理。
- 复现实验：基线结果与样本均已随仓库提供，可直接使用现有 CSV 进行下游实验，无需重新下载。
