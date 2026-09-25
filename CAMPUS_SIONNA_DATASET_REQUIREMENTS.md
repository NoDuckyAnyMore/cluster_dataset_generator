# 高校 Sionna RT 三维体素数据集要求

> 当前集群存储规范（2026-09-03）：ROOT 为 `~/vast/UAV_RM`，数据集名为
> `project985_39_main_voxel_256m_128x128x40_rxexpand_float32_rand10to32`。
> 每个 RX 直接保存一个 `float32`、shape `[Z,Y,X]=[40,128,128]` 的
> `sionna_results_by_rx/rx_NNN.npy`。每批结果写入 `.partial.npy` memmap，
> flush 后更新 checkpoint，全部完成才原子改名并提交 `index.json`；不再生成或拆分 CSV。
> 下文若仍出现 compact5/CSV，均只表示旧数据兼容格式，不是当前生成格式。

本文档同时记录原深圳高校数据集和新的 39 所 985 主校区数据集。当前新增数据由Slurm单卡任务数组生成；每个数组任务是完全相同的独立worker。

## 数据集范围

### 39 所 985 主校区数据集（当前方案）

- 学校共 39 所：北京大学、清华大学、中国人民大学、北京航空航天大学、北京理工大学、中国农业大学、北京师范大学、中央民族大学、南开大学、天津大学、大连理工大学、东北大学、吉林大学、哈尔滨工业大学、复旦大学、同济大学、上海交通大学、华东师范大学、南京大学、东南大学、浙江大学、中国科学技术大学、厦门大学、山东大学、中国海洋大学、武汉大学、华中科技大学、湖南大学、中南大学、中山大学、华南理工大学、四川大学、重庆大学、电子科技大学、西安交通大学、西北工业大学、西北农林科技大学、兰州大学、国防科技大学。
- 校园边界采用各校主校区的 OSM 边界；厦门大学使用已核对的主校区 OSM way，而不是名称检索误命中的小学边界。
- 地图共 `1436` 个与校园边界相交的 `256 m × 256 m` 区块。
- 全部39校交给RTX 5090集群。默认提交4个独立单卡worker；各任务分别排队，通过逐block文件锁动态领取工作。每个worker使用一个长期存活的Python/Sionna进程连续处理多个block，避免block之间反复冷启动。改变worker数量只改Slurm数组范围，不修改Python。
- 集群 ROOT：默认 `~/vast/UAV_RM`，可由环境变量 `RID_CLUSTER_ROOT` 覆盖。
- 启动器：`cluster_dataset_generator/run_cluster_5090.py`，通过 `sbatch submit_5090.slurm` 提交。
- 数据集目录：`~/vast/UAV_RM/project985_39_main_voxel_256m_128x128x40_rxexpand_float32_rand10to32/`。
- 全局随机种子：`20260921`。

### 深圳高校数据集（已有独立数据集）

- 包含南方科技大学、港中深、哈工深、北大深研院、清华深研院、深圳大学粤海校区、深圳大学丽湖校区。
- 它与 39 校 985 数据集是两个独立数据集，不要求与新 985 数据合并。

## 共同空间与收发设置

- OSM 地图按 `256 m × 256 m` 切块，只保留与校园边界相交的块。
- 每块建立 `128 × 128 × 40` 的空中 TX 体素网格，体素边长 `2 m`。
- 水平体素中心坐标为 `-127,-125,…,127 m`；高度为 `2,4,…,80 m`。
- 每块默认生成 10 个地面 RX，但 RX 数量可扩容；RX 高度为 `0.5 m`，与建筑平面轮廓的最小距离为 `5 m`。
- RX 采用与目标总数无关的确定性随机序列。把 RX 数量从 N 增至 M 时，前 N 个位置不变，只新增索引 N 至 M-1。
- 默认每块结果为 `10 × 128 × 128 × 40 = 6,553,600` 行；每新增一个 RX 增加 `655,360` 行。
- 每块 OSM、RX 位置、体素索引和 metadata 足以重建 TX/RX 三维坐标，不在结果 CSV 中逐行重复这些常量。

## 建筑高度

- OSM 有有效 `height` 时直接使用。
- 没有 `height`、但有 `building:levels` 或 `levels` 时，按 `3.2 m/层` 计算。
- 两者都没有时，给全部未知高度建筑赋值 `12 m + U[-2,20] m`，即 `10–32 m`。
- 39 校 985 随机高度缓存使用种子 `20260921`，随机比例为 `100%`。
- 随机高度由全局种子和 OSM way ID 确定；同一建筑跨块出现时高度保持一致。
- 随机高度仅用于仿真多样性，不作为真实环境 GT。

## Sionna RT 参数

- 频率：`2.437 GHz`；发射功率：`20 dBm`。
- TX/RX 天线：`iso`、V 极化。
- `max_depth=3`，`samples_per_tx=10000`。
- 启用 LOS、镜面反射、折射和绕射；关闭漫反射。
- 建筑和地面使用 ITU concrete，厚度 `0.3 m`。
- 39 校 985 的 RTX 5090 专用 `TX_BATCH_SIZE=130`。
- 每个 5090 默认使用 `CPU_THREADS_PER_WORKER=8`；每张 Slurm 分配的 GPU 运行一个独立 worker。

## 按 RX 分片的 compact5 结果 CSV

- 新仿真固定输出以下 5 列：

```text
rx_index,tx_voxel_ix,tx_voxel_iy,tx_voxel_iz,rss_dbm
```

- 区块身份来自目录；坐标和 RX 位置来自 `metadata.json` 与 `rx_positions.csv`；RT 常量来自 metadata。
- 不再把整个 block 写入一个巨型 CSV。结果保存为 `sionna_results_by_rx/rx_000.csv`、`rx_001.csv`……，每个文件固定 `655,360` 行，并由 `index.json` 和 `groups/*.complete.json` 原子确认完成状态。
- 求解时默认把10个缺失RX放在同一组计算，以保留多RX求解效率；求解结束后再拆成每RX一个文件。`RX_SOLVER_GROUP_SIZE` 在Python启动器顶部设置，只影响显存和吞吐，不改变既有结果兼容性。
- 扩容时已有 RX 文件不会重算，只补算缺失索引；不允许在原数据集目录内缩小 RX 数量。
- 旧 27 列结果使用 `convert_results_csv_to_compact5.py` 转换。转换器默认使用 12 个进程。
- 每个 RX 分片必须通过 655,360 行、索引范围、RSS 有限性以及全部 `[Z,Y,X]` 唯一组合校验。
- 只有临时文件完整校验并原子安装成功后，才删除旧 27 列备份；失败时保留或恢复旧文件。
- compact5 训练预处理器必须从体素索引和 metadata 重建坐标；不能依赖已删除的 27 列重复字段。

## 128 × 128 RSS 预览图

- 39 校 985 启动器默认设置 `RENDER_PREVIEW_AFTER_BLOCK=False`，避免在 JuiceFS HDD 上创建 574,400 个小文件并占用 GPU 作业时间。需要时独立生成少量预览，不重新运行 Sionna。
- 新分片图片程序为 `render_rx_sharded_channel_maps.py`；旧单 CSV 数据仍由 `render_voxel_channel_maps.py` 兼容处理。
- 路径和命名规则保持不变：

```text
<数据集>/<学校>/<block>/rss_maps_128px/rx_000/z_000_2m.png
...
<数据集>/<学校>/<block>/rss_maps_128px/rx_009/z_039_80m.png
```

- 默认每个 block 生成 `10 RX × 40 高度 = 400` 张严格 `128 × 128` PNG；RX 扩容后图片数量自动变为 `RX 数量 × 40`。
- OSM 建筑体在当前高度切片内的像素覆盖为纯黑色 `#000000`：仅当楼顶高度不低于当前 TX 层时涂黑，因此矮楼不会继续出现在 80 m 切片中；Sionna 无解析路径的像素显示为灰色 `#d1d5db`。
- RSS 色彩范围仅用于预览显示，不定义训练归一化规则。
- 完成标记同时绑定 CSV、OSM 和 renderer 版本。旧版图需要整批刷新；同一新版任务中断后只补缺失图片。
- 自动绘图不创建单独的持久化log；进度和错误写入对应的 `generation_worker_<job>_<task>.log`。

## OSM 缓存与目录

- 39 校原始 OSM 缓存：`cluster_dataset_generator/offline_osm_cache_985_256m/`。
- 39 校随机高度 OSM 缓存：`cluster_dataset_generator/osm_randomized_height_985_256m_u10_32/`。
- 原始缓存包含 manifest、39 份校园边界和 1436 份有效 OSM；随机缓存只包含与 manifest 一一对应的 1436 份 `osm_map.osm`。
- 深圳原始缓存仍为 `offline_osm_cache_shenzhen_256m/`；深圳随机高度缓存仍为 `osm_randomized_height_shenzhen_256m_u10_32/`。
- 不下载或复制卫星图。
- `cluster_dataset_generator` 与最终数据集目录位于同一个 ROOT 下，彼此为同级目录。

## 断点、恢复与验收

- 每个空中 TX batch 写入当前 RX 组的临时 CSV 并刷新 checkpoint；整组完成、拆分且逐 RX 校验后，才原子写入组完成标记。
- compact5 checkpoint 使用独立 schema 签名，禁止把旧 27 列临时结果与新 5 列结果混写。
- batch 是运行时参数，不属于物理仿真签名；修改 batch 后可从当前未完成 block 已确认的绝对 TX 偏移继续，完整 block 不重算。
- 若 RT 已完成但预览中断，重新提交集群作业时只继续预览后处理，不重新计算 RT。
- 最终使用 `audit_campus_sionna_dataset.py --quick` 检查动态 RX 数量、分片完成标记、科学参数、compact5 schema 和预览完成标记；去掉 `--quick` 才逐行检查全部 RX 分片。
