# RID/Sionna 集群信道地图数据生成器

本项目从 OpenStreetMap 建筑轮廓构建 Sionna RT 场景，在 Slurm 集群上生成无人机三维发射体素与地面接收机之间的 RSS 信道地图。当前主配置覆盖 39 所 985 高校主校区，支持 RTX 4090/5090 混合计算、动态领取区块、断点续算和按接收机分片保存结果。

## 主要输出

- 地图按 `256 m x 256 m` 划分区块。
- 每个区块包含 `128 x 128 x 40` 个发射体素，体素边长为 2 m，高度为 2–80 m。
- 默认生成 10 个地面接收机；每个接收机保存一个 `float32 [40,128,128]` NPY。
- 缺失的 OSM 建筑高度可使用固定随机种子生成 10–32 m 的合成高度。合成高度只用于仿真多样性，不是真实环境真值。

## 环境

推荐 Ubuntu、Slurm、NVIDIA GPU/CUDA 和 Conda。Python 依赖见 `requirements-ubuntu.txt`：Sionna RT、Mitsuba、Dr.Jit、NumPy 与 Matplotlib。

```bash
git clone https://github.com/NoDuckyAnyMore/cluster_dataset_generator.git
cd cluster_dataset_generator
bash setup_cluster_env.sh
```

集群的 module 名称、CUDA 版本和 Slurm 分区可能需要按实际平台修改 `setup_cluster_env.sh`、`submit_5090.slurm` 与 `submit_4090.slurm`。

## 准备 OSM 缓存

OSM 缓存和生成结果体积较大，不包含在 Git 仓库中。请在可以联网的节点生成：

```bash
conda activate UAV_RM
python cache_985_osm_local_256m.py
python randomize_missing_osm_heights.py \
  --input-root offline_osm_cache_985_256m \
  --output-root osm_randomized_height_985_256m_u10_32 \
  --seed 20260921 \
  --fraction 1.0
```

计算节点不联网时，应先将代码以及上面两个缓存目录同步到集群工作目录。

## 提交计算

默认结果根目录为 `~/vast/UAV_RM`，可通过 `RID_CLUSTER_ROOT` 覆盖。

2026-10-02 新批次只更新地面 RX 位置：默认 `RID_RANDOM_SEED=20261002`，继续使用原有
`osm_randomized_height_985_256m_u10_32` 缓存及其建筑高度种子 `20260921`。
结果目录自动加上 `_rxseed20261002`，默认计算全部 39 所高校、1,436 个区块，不使用旧跳过名单。
缓存已在集群时，无需重新运行上面的缓存准备命令。

提交 8 个 RTX 5090 单卡 worker：

```bash
RID_RANDOM_SEED=20261002 RID_SKIP_FILE= sbatch --array=0-7%8 submit_5090.slurm
```

以后只需修改 `RID_RANDOM_SEED` 就会自动选择对应 RX 种子的独立结果目录。
`RID_DATASET_NAME` 可自定义目录名，`RID_SKIP_FILE` 可显式启用某批次专用的外部完成名单。
根目录 `dataset_metadata.json` 的 `simulation_defaults.random_seed_global` 和各区块
`metadata.json` 的 `random_seed_global`、`random_seed_derived` 记录 RX 种子；
`osm_height_variant` 单独记录沿用的建筑高度版本。

```bash
# RTX 5090
sbatch submit_5090.slurm

# RTX 4090
sbatch submit_4090.slurm
```

常用覆盖参数：

```bash
RID_CLUSTER_ROOT=/path/to/workspace \
RID_CPU_THREADS_PER_GPU=8 \
RID_TASK_ORDER=simple_first \
sbatch submit_5090.slurm
```

查看进度与执行快速审计：

```bash
python3 show_dataset_progress.py --by-region
python audit_campus_sionna_dataset.py --quick \
  --dataset-root "${RID_CLUSTER_ROOT:-$HOME/vast/UAV_RM}/project985_39_main_voxel_256m_128x128x40_rxexpand_float32_rand10to32_rxseed20261002"
```

运行不依赖 GPU 的单元测试：

```bash
python -m unittest discover -s tests -v
```

更完整的集群配置、恢复流程、结果格式和审计说明见 [README_CLUSTER.md](README_CLUSTER.md) 与 [CAMPUS_SIONNA_DATASET_REQUIREMENTS.md](CAMPUS_SIONNA_DATASET_REQUIREMENTS.md)。

## 分卷导出数据集

使用独立的 Conda 环境 `archive_tools` 安装 `7zip`，运行命令为 `7zz`。
20261002 RX 批次使用 2 个压缩线程、每卷 8 GiB，保存到 `~/vast/UAV_RM/export_20261006`。
安装、代理超时处理、压缩及解压命令见
[集群 7-Zip 分卷导出教程](README_CLUSTER.md#使用-7-zip-分卷导出数据集)。

## 数据说明

仓库只发布生成代码和文档，不发布下载的 OSM 缓存、卫星图、仿真张量、checkpoint 或集群日志。使用或再分发 OpenStreetMap 派生数据时，请遵守 ODbL 并保留 OpenStreetMap contributors 署名。
