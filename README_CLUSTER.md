# RID/Sionna RTX 5090 集群生成器

这是 `ubuntu_dataset_generator` 的独立集群副本。原课题组服务器代码不会被本目录影响。

## 一条命令查看 GPU 卡时、费用和 VAST 用量

上传 `cluster_usage.sh` 到本目录，在登录节点执行一次：

```bash
cd "$HOME/vast/UAV_RM/cluster_dataset_generator"
bash cluster_usage.sh --install
```

脚本按当前登录 shell 将下面的环境变量和函数追加到 `.bashrc` 或 `.zshrc`，不需要 sudo；
重复安装不会重复追加。随后执行安装提示中的 `source` 命令，或重新登录，即可在任意目录输入：

```bash
cluster_usage
```

配置内容等价于：

```bash
export CLUSTER_USAGE_SCRIPT="$HOME/vast/UAV_RM/cluster_dataset_generator/cluster_usage.sh"
cluster_usage() { bash "$CLUSTER_USAGE_SCRIPT" "$@"; }
```

只需 Bash、awk、GNU coreutils 和 Slurm 的 `sacct`，不需要 Python、Conda、module load、
GPU 作业或执行权限（通过 `bash` 调用）。脚本无持久日志，也不修改数据集、checkpoint 或作业。

- 默认查询当前用户所有 Slurm 可用历史，按北京时间把跨日作业拆到每天；没有用量的天不输出。
- 若涉及两个及以上有用量的自然月，额外输出逐月汇总；始终输出总卡时和总费用。
- 5090 按 **2.70 元/卡时**，4090 按 **2.16 元/卡时**；同时显示两种卡各自卡时和费用。
- 一个任务多卡时乘以实际分配卡数；数组 worker 独立统计，排除 `.batch`/`.extern` 重复计数。
- 运行中以及失败、取消任务实际运行过的时间都计入；排队不计。不同数据库作业记录保留，
  重复记录去重。历史被管理员清理后无法还原；重排队/扩缩容等复杂记账情况以平台账单为准。
- 对运行秒数与起止跨度不同的记录（例如暂停），按有效运行秒数比例分摊到各日并提示。
  `sacct` 汇总无法提供暂停的具体日期，因此这类记录的每日费用只是估算。
- 其他 GPU 型号计入 `Other(h)` 和总卡时，但没有价格，不会假装计入其费用。
- VAST 依次显示 `df -hT`、`quota -s`、`du -sh`。共享文件系统的剩余量、个人配额、
  个人目录大小是三个不同指标。未设置存储单价，因此这里不估算存储费用。
- Slurm 查询超时/失败时明确显示未知，不输出零费用冒充成功；仍会继续查询存储。

可选用法（结束时间为不包含的上界）：

```bash
cluster_usage --since 2026-09-01
cluster_usage --since 2026-09-01 --until 2026-10-01
cluster_usage --no-du
```

可在 shell 配置中设置 `UAV_USAGE_START` 改默认起始日期，`UAV_VAST_DIR` 改 VAST 目录
（默认 `$HOME/vast`），`UAV_USAGE_TZ` 改时区，`UAV_RATE_5090` / `UAV_RATE_4090` 改价格。
移动脚本后更新 `CLUSTER_USAGE_SCRIPT` 即可。安装时 `UAV_USAGE_RC` 可指定其他 shell 配置文件。

记账字段参考：[Slurm sacct 官方文档](https://slurm.schedmd.com/sacct.html)。

## 默认任务

- Slurm 队列：`gpu_5090`
- 默认提交：4个彼此独立的Slurm数组任务，每个任务只申请1张RTX 5090
- 四个worker不需要同时获得GPU；调度器有一张空闲卡就可以先启动一个
- 所有worker完全相同，通过逐block文件锁动态领取下一个任务
- 每个worker只启动一次Python/Sionna进程，并在同一进程内连续领取多个block；block之间不再用子进程重启，因此Dr.Jit/Mitsuba/Sionna的内存JIT缓存可继续复用
- 中断后重新提交同一个任务，已完成 block 会跳过，未完成 block 从 batch checkpoint 续跑
- 每个block只有一个worker能持有锁；进程退出时锁自动释放，其他worker可从checkpoint接手
- 计算节点不联网；39 所 985 主校区的原始及随机高度 OSM 已随目录打包
- 默认 RX 种子：`20261002`；默认计算全部 1,436 个区块，不启用外部跳过名单
- 默认结果目录：`~/vast/UAV_RM/project985_39_main_voxel_256m_128x128x40_rxexpand_float32_rand10to32_rxseed20261002`
- 原有 OSM 缓存及建筑高度版本（种子 `20260921`）继续使用；新批次只重新采样地面 RX
- 每个 RX 独立保存为一个 float32 `[40,128,128]` NPY；后续增加 RX 数量时只计算新增 RX
- 默认关闭逐层 PNG 预览，避免在 JuiceFS HDD 上创建几十万个小文件；需要时可在 RT 完成后独立绘制
- 默认按估算建筑三角面数从少到多领取 block（`simple_first`），先运行简单场景；排序只影响领取顺序，不影响结果或 checkpoint

## 混合使用5090和4090

`run_cluster_5090.py` 现在是两种GPU共用的长期worker。GPU型号和worker数量由Slurm提交脚本决定，科学参数、动态block锁及NPY格式保持一致：

- `submit_5090.slurm`：分区 `gpu_5090`，默认 `TX_BATCH_SIZE=120`。
- `submit_4090.slurm`：分区 `gpu_4090`，默认 `TX_BATCH_SIZE=60`。
- 两种作业可同时运行并领取同一数据集。batch大小不是物理结果签名的一部分，因此一个block可由4090从5090的checkpoint续跑，反向也一样。

默认提交4个4090单卡worker：

```bash
sbatch submit_4090.slurm
```

先测试一个或临时改变数量时，无需修改Python：

```bash
sbatch --array=0-0%1 submit_4090.slurm
sbatch --array=0-5%6 submit_4090.slurm
```

若某张4090仍因复杂场景显存不足，可在提交时进一步降低batch，例如：

```bash
RID_TX_BATCH_SIZE=50 sbatch --array=0-0%1 submit_4090.slurm
```

## 跳过已在其他服务器完成的学校

`completed_regions_skip.txt` 记录已经在独立 5090 服务器完整计算并核验的
旧批次 `region_slug`。新 RX 种子批次默认不读取此清单，全部重新计算。
只有显式设置 `RID_SKIP_FILE=/path/to/batch_skip.txt` 时，worker 才读取指定清单，
并在领取 block 之前排除这些学校。旧清单包含 9 所整校完成的大学，共 224 个 block；
南开和同济没有整校完成，因此不在清单中。

清单只控制集群是否计算，不会凭空复制结果。最终合并数据集时，仍需把这些
学校的完整目录复制到集群数据集目录；最终审计始终检查完整的 1436 个 block，
不会因为跳过清单而提前宣布完成。以后需要恢复某所学校的集群计算时，删除
清单里的对应行并重新提交任务即可。

## NPY 数组与写入顺序

每个 `sionna_results_by_rx/rx_NNN.npy` 都是一个 `float32` 三维数组，shape 为：

```text
[Z, Y, X] = [40, 128, 128]
```

- 第一维 `Z` 是 TX 高度层：索引 `0..39` 对应 `2, 4, ..., 80 m`。
- 第二维 `Y` 是南北方向的水平网格索引：从局部坐标 `-127 m` 递增到 `+127 m`。
- 第三维 `X` 是东西方向的水平网格索引：从局部坐标 `-127 m` 递增到 `+127 m`。
- 每个高度平面按 `Y` 外层、`X` 内层生成，因此 `X` 变化最快。平面展平索引为 `flat_index = iy * 128 + ix`。
- 求解顺序是先完成当前高度的全部水平点，再进入下一高度；一个 batch 对应当前高度展平数组中的连续区间。
- 同一 RX 的 batch 直接写入 `tensor[iz].reshape(-1)[batch_start:batch_end]`；不同 RX 分别写入自己的 NPY，不混在同一个文件中。
- 每个 batch 先写入内存映射的 `.partial.npy`；默认在每第25个batch和每层最后一个batch统一 flush 所有RX，再更新checkpoint。checkpoint只会指向已经flush的数据。异常中断后从最后一个持久化边界续跑，最多重算24个batch；已经确认的数据位置以及跨batch大小续跑能力不变。
- 日志每个高度打印第1个、每第25个以及最后一个batch的进度。持久化间隔由 `run_cluster_5090.py` 中 `CHECKPOINT_INTERVAL_BATCHES=25` 设置，只影响写盘频率和中断后最多重算量，不改变数值结果或结果签名。
- `run_cluster_5090.py` 中 `ENABLE_STAGE_TIMING=True` 时，每完成一个高度会额外输出一行 `HEIGHT TIMING`，分别统计 `tx_setup`、`solver_call`、`cir_numpy_sync`、`gain_reduce`、`tensor_write_stats`、`npy_flush`、`checkpoint` 和 `tx_cleanup`。其中 `cir_numpy_sync` 包含尚未完成的 CIR 工作、GPU 同步及 GPU 到 CPU 的 NumPy 传输，不能解释为纯 PCIe 通信时间。计时不开启额外 GPU 同步，不改变数值结果或 checkpoint；不需要时将开关改成 `False`。
- `ENABLE_DRJIT_KERNEL_HISTORY=True` 时，使用Dr.Jit官方 `JitFlag.KernelHistory` API，每完成一个高度输出一行 `DRJIT KERNEL HISTORY`。其中 `memory_hits` 是进程内JIT缓存命中，`disk_hits` 是磁盘缓存命中，`hard_misses` 是需要重新编译的kernel；`execution_time_ms`、`codegen_time_ms`、`backend_time_ms` 是本层总计。`miss_codegen_time_ms`、`miss_backend_time_ms` 和二者之和 `miss_compile_time_ms` 明确记录真正miss造成的编译开销；`miss_execution_time_ms` 与 `hit_execution_time_ms` 分别统计miss和命中kernel的设备执行时间。历史在每个高度后读取并清空，不会跨高度重复累计。
- 不启用逐kernel的原始Info刷屏；一个block包含数千次kernel启动，原始输出会令日志膨胀。官方KernelHistory的逐高度汇总会同时写入Slurm `.out` 和数据集根目录下的 `generation_worker_<job>_<task>.log`，GPU/OptiX预检输出也会进入该worker日志。
- `ENABLE_GPU_MEMORY_LOG=True` 时，每完成一个block调用一次 `nvidia-smi`，输出 `GPU MEMORY`，包括本worker进程显存、整卡已用/空闲/总显存和GPU UUID。它只读采样，不清理Dr.Jit malloc或kernel缓存；每block一次的开销远小于逐batch或逐高度采样。
- `ENABLE_BLOCK_MEMORY_RECLAIM=True` 时，每个完整block结束后依次执行 Python 垃圾回收、`dr.sync_thread()` 和 `dr.flush_malloc_cache()`，并输出回收前后两行 `GPU MEMORY` 与一行 `DRJIT MEMORY RECLAIM`。它只归还已经不用的分配缓存；不会调用 `dr.flush_kernel_cache()`，所以进程内已编译kernel和磁盘缓存继续保留。
- 40 个高度全部完成并通过 shape、dtype、有限值检查后，`.partial.npy` 才原子改名为 `rx_NNN.npy`，随后更新 `index.json`。

读取某个体素的 RSS：

```python
rss_dbm = np.load("rx_000.npy", mmap_mode="r")[iz, iy, ix]
```

局部坐标可由 metadata 中的体素尺寸重建；当前固定参数下：

```text
x_m = -128 + (ix + 0.5) * 2
y_m = -128 + (iy + 0.5) * 2
z_m = 2 + iz * 2
```

## 推荐上传位置

把整个目录上传为：

```text
~/vast/UAV_RM/cluster_dataset_generator
```

生成器与数据集目录最终并列：

```text
~/vast/UAV_RM/
├── cluster_dataset_generator/
└── project985_39_main_voxel_256m_128x128x40_rxexpand_float32_rand10to32_rxseed20261002/
```

如果需要改变根目录，在提交时设置：

```bash
RID_CLUSTER_ROOT="$HOME/vast/UAV_RM_alt" sbatch submit_5090.slurm
```

ZIP 解压后先检查目录是否可写。某些打包工具会把目录权限保存成只读，登录
节点虽然能 `cd` 和读取文件，但计算节点无法创建 Slurm 输出、锁文件和结果：

```bash
chmod u+rwx "$HOME/vast/UAV_RM"
chmod -R u+rwX "$HOME/vast/UAV_RM/cluster_dataset_generator"
cd "$HOME/vast/UAV_RM/cluster_dataset_generator"
touch .write-test && rm .write-test
ls -ld .
```

最后一条的权限中，目录所有者应包含 `rwx`，不能是之前遇到的
`dr-xr-xr-x`。如果数据集目录已经存在，也要保证它可写：

```bash
mkdir -p "$HOME/vast/UAV_RM/project985_39_main_voxel_256m_128x128x40_rxexpand_float32_rand10to32_rxseed20261002"
chmod -R u+rwX "$HOME/vast/UAV_RM/project985_39_main_voxel_256m_128x128x40_rxexpand_float32_rand10to32_rxseed20261002"
```

## 提交和续跑

登录节点只负责提交，不能直接运行 Python：

```bash
cd ~/vast/UAV_RM/cluster_dataset_generator
sbatch submit_5090.slurm
```

`submit_5090.slurm` 使用 Slurm 提交目录 `SLURM_SUBMIT_DIR` 寻找生成器，因此
必须先进入生成器目录再提交；不要从其他目录用相对路径提交。

第一次上传新版本时，建议先只启动一个 worker 验证环境、权限和显存：

```bash
sbatch --array=0-0%1 submit_5090.slurm
squeue -u "$USER"
```

记下返回的作业号后实时查看输出：

```bash
tail -f "slurm-Sionna985-5090-作业号_0.out"  # 5090
tail -f "slurm-Sionna985-4090-作业号_0.out"  # 4090
```

现在 GPU/OptiX 预检会同时输出到 Slurm 日志和数据集根目录的worker日志。正常时会依次看到
`Checking allocated GPU slot 0`、版本信息、`OptiX scene creation OK`，然后
进入 `WORKER ... CLAIM` 和 block 计算。确认一个 worker 正常后，可以保留它，
再提交三个相同 worker：

```bash
sbatch --array=0-2%3 submit_5090.slurm
```

也可以取消单 worker 测试，再直接使用文件中的默认四 worker 配置。

查看作业：

```bash
squeue -u "$USER"
parajobs
```

查看整个数据集的实时完成数量、剩余数量和可续跑 checkpoint 数量（不会启动
Sionna，也不需要 GPU）：

```bash
python show_dataset_progress.py
```

按学校展开：

```bash
python show_dataset_progress.py --by-region
```

每 30 秒自动刷新：

```bash
watch -n 30 python show_dataset_progress.py
```

这里的 `External skip blocks` 是已在其他服务器完成、由
显式设置的 `RID_SKIP_FILE` 排除的任务（默认没有）；`Cluster complete` 是当前集群数据
目录中通过完整 metadata、index 和 10 个 RX NPY 检查的 block；两者共同计入
`Task progress`。

取消作业：

```bash
scancel 作业ID
```

被取消、超时或节点中断后，重新执行 `sbatch submit_5090.slurm` 即可续跑。

查看已结束任务的状态和退出码：

```bash
sacct -j 作业号 --format=JobID%20,State%15,ExitCode,Elapsed,MaxRSS,NodeList%20
scontrol show job 作业号_数组序号
```

## 改变worker/GPU数量

本次新 RX 批次使用 8 张 RTX 5090，无需修改 Slurm 文件：

```bash
RID_RANDOM_SEED=20261002 RID_SKIP_FILE= sbatch --array=0-7%8 submit_5090.slurm
```

后续更换 RX 种子仍可通过 `RID_RANDOM_SEED` 指定，结果目录自动加 `_rxseed<种子>`。
建筑高度缓存与 RX 种子独立，继续复用现有缓存。

只修改 `submit_5090.slurm` 的数组范围，不修改Python。例如默认4个单卡worker：

```text
#SBATCH --array=0-3%4
#SBATCH --gpus=1
```

改成6个单卡worker：

```text
#SBATCH --array=0-5%6
#SBATCH --gpus=1
```

也可不改文件，提交时临时覆盖：`sbatch --array=0-5%6 submit_5090.slurm`。每个数组子任务独立排队，空出几张卡就运行几个。

每卡CPU线程默认8，可通过下面的方式调整：

```bash
RID_CPU_THREADS_PER_GPU=6 sbatch submit_5090.slurm
```

默认任务顺序为简单场景优先。无需修改 Python 即可在提交时切换：

```bash
# 默认：建筑三角面少的block先运行
RID_TASK_ORDER=simple_first sbatch submit_5090.slurm

# 建筑三角面多的block先运行
RID_TASK_ORDER=complex_first sbatch submit_5090.slurm

# 完全保留blocks_manifest.csv原始顺序
RID_TASK_ORDER=manifest sbatch submit_5090.slurm
```

改变顺序不会使任何结果失效。复杂场景已有的 batch checkpoint 会保留，等
worker 后续领取到对应 block 时从原偏移继续。

## 改变仿真参数或扩容RX

这些设置只在 `run_cluster_5090.py` 顶部修改，不写进Slurm文件：

```python
RX_COUNT = 10
RX_SOLVER_GROUP_SIZE = 10
TX_BATCH_SIZE = 120
```

例如把 `RX_COUNT` 从10改成16后重新提交，前10个RX位置和结果不变，只补算 `rx_010.npy` 至 `rx_015.npy`。修改RX目标前先取消仍在运行的旧worker，避免两套目标同时操作同一数据集。

求解组大小不是物理参数，不会让已完成分片失效。但一个尚未完成的 RX 组若改变组大小，该组的临时 checkpoint 不能复用；已提交的 RX 分片仍会保留。

`TX_BATCH_SIZE` 也不是数据定义的一部分。当前 batch 为 120；改变它后，已经
由 checkpoint 确认的体素不会重算，下一批从记录的展平位置继续，最终 NPY
与使用哪种 batch 切分无关。`SAMPLES_PER_TX` 和 `MAX_DEPTH` 是仿真物理/精度
设置，不应为了显存问题随意更改，否则会产生口径不同的数据。

## 本次集群问题与代码修复

### 作业瞬间退出且没有 Slurm 日志

曾出现 `FAILED 0:53`、运行时间 `00:00:00`，同时找不到 `.out` 文件。最终确认
不是 Python 或 Sionna 报错，而是生成器目录权限为 `dr-xr-xr-x`，计算节点
无法创建 stdout。按“推荐上传位置”中的 `chmod` 和 `touch` 检查处理即可。

- `ExitCode=0:53` 且没有应用日志：优先检查工作目录、输出路径及写权限。
- `ExitCode=1:0` 且已有日志：说明脚本已经启动，应查看 traceback 的最后部分。

### RTX 5090 在第二批附近显存溢出

曾出现：

```text
jit_malloc(): out of memory! Could not allocate 2147483648 bytes of device memory
```

原因不是该 block 建筑复杂，也不是 NPY 占用显存。新 NPY 求解循环之前没有在
下一次 `solver(...)` 调用前释放上一批的 Sionna `Paths` 对象；Python 会先计算
赋值号右侧，因此新旧两个大型路径缓冲短暂同时驻留显存。代码现已在每批开始
前以及 `finally` 中显式释放 `paths/a/tau/gains/powers`，包括异常路径。

当前使用 `TX_BATCH_SIZE=120`、`SAMPLES_PER_TX=10000`、`MAX_DEPTH=3`
等仿真口径。若上传本版本后仍发生真实 OOM，才把 `run_cluster_5090.py` 的
`TX_BATCH_SIZE` 逐步降到 120、100；现有 batch checkpoint 可以继续使用。

### 多个worker同时启动时报OptiX磁盘缓存数据库错误

并发启动多个Slurm数组任务时曾出现：

```text
OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR (7012)
jit_optix_log(): [DISKCACHE] Error when configuring the database
```

原因是多个worker同时使用网络家目录中的同一个OptiX磁盘缓存数据库。现在每
个数组worker都会在节点本地的 `SLURM_TMPDIR`、`TMPDIR` 或 `/tmp` 下创建唯一
的 `uav_rm_<array-job>_<task>/optix_cache` 和 `cuda_cache`，并分别设置
`OPTIX_CACHE_PATH`、`CUDA_CACHE_PATH`。这些缓存不属于数据集，节点清理后可以
丢弃；日志会打印 `OptiX cache path` 以便核验。NVIDIA OptiX官方也要求缓存
路径有效且可写，并支持通过 `OPTIX_CACHE_PATH` 覆盖默认位置。

### 旧 RX 坐标被误判为位置变化

旧版 `rx_positions.csv` 只保存三位小数，但续跑时按 `1e-6 m` 与内存中的完整
坐标比较，最大约 `0.0005 m` 的正常四舍五入误差会触发：

```text
Deterministic RX prefix changed at index 0
```

新版文件保存六位小数，并允许旧三位小数文件的最大舍入误差。因此旧 RX、旧
checkpoint 和已经完成的 NPY 均可直接续用，不要删除或重新生成。

### 大 OSM 文件不等于复杂建筑场景

之前 worker 按 OSM 文件字节数从大到小领取任务。两个最先失败的西农 block
实际都是 0 栋建筑、场景中只有地面，却因 OSM 内含大型道路 route relation
而各有约 1.6 万个 relation member，文件异常大。这种排序既不能代表 Sionna
网格复杂度，也会误导排障。

现在任务默认按 OSM 建筑 footprint 估算出的网格面数、建筑数从少到多排序，
文件体积只作最后的同分项。道路 relation 不再被当成建筑复杂度；也可通过
`RID_TASK_ORDER` 切换成复杂优先或 manifest 原顺序。该估算只用于调度顺序，
不会改变场景解析、仿真结果或断点格式。

## 查看5090资源和空闲情况

查看5090队列每个节点配置、状态及已用GRES：

```bash
sinfo -p gpu_5090 -N -O NodeList:20,StateCompact:12,Gres:35,GresUsed:35
```

如果该集群的Slurm版本不支持 `GresUsed` 字段：

```bash
sinfo -p gpu_5090 -N -o "%N %t %G"
squeue -p gpu_5090 -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R %b"
```

进一步查看某节点的总GPU与已分配GPU：

```bash
scontrol show node 节点名 | grep -E "State=|CfgTRES=|AllocTRES=|Gres="
```

其中空闲GPU数量约等于 `CfgTRES` 中的 GPU 总数减去 `AllocTRES` 中的 GPU 数量。登录节点上直接运行 `nvidia-smi` 不能代表调度队列还有多少空闲卡。

## 查看空间

`df` 显示共享文件系统整体容量，并不一定是你的个人配额：

```bash
df -hT "$HOME/run/" "$HOME/ssd/" "$HOME/vast/"
```

查看自己实际已经占用多少，以及各一级目录大小：

```bash
du -sh "$HOME/run/" "$HOME/ssd/" "$HOME/vast/" 2>/dev/null
du -sh "$HOME/vast"/* 2>/dev/null | sort -h
```

优先用下面的命令查个人限额和剩余配额：

```bash
quota -s
```

如果 `quota` 没有信息，说明集群可能使用独立存储配额工具，需要向管理员查询；此时不能把 `df` 的剩余空间当成个人可用空间。

## Conda环境

先在可以联网的登录节点安装环境（不要在登录节点运行仿真）：

```bash
module load miniforge3/26.3.2-3
module load cuda/12.8
bash setup_cluster_env.sh
```

环境脚本和 Slurm 提交脚本都会显式加载相同模块，登录节点当前是否已经 `module load` 不影响计算节点。

提交脚本默认激活 `UAV_RM`。如果环境名称不同：

```bash
RID_CONDA_ENV=你的环境名 sbatch submit_5090.slurm
```

主程序是 `run_cluster_5090.py`，但不要在登录节点直接执行它；必须通过 `sbatch` 进入计算节点。
