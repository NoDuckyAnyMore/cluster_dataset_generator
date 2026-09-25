# 高校 Sionna RT 三维体素数据集状态

这是随集群生成器提供的初始说明。实际运行后，主程序会在数据集根目录重新生成 `RESULTS_REPORT.md`。

- 地图分块：256 m × 256 m
- 空中 TX 体素：128 × 128 × 40，体素边长 2 m，高度 2–80 m
- 地面 RX：默认 10 个，采用确定性随机序列，可在原目录中向上扩容
- 结果格式：每个 RX 一个 float32 `[40,128,128]` NPY
- 自动预览：每个 RX、每个高度一张 128×128 PNG

## 单个 block 的主要文件

- `metadata.json`
- `osm_map.osm`
- `osm_buildings_2d.png` / `osm_buildings_3d.png`
- `rx_positions.csv` / `rx_positions_2d.png`
- `scene_generated/`
- `sionna_results_by_rx/rx_NNN.npy`
- `sionna_results_by_rx/index.json`
- `sionna_results_by_rx/.work/*.checkpoint.json`（仅未完成时存在）
- `rss_maps_128px/rx_NNN/z_ZZZ_HEIGHTm.png`

中断后，同一 RX 求解组从 batch checkpoint 续跑；已经通过校验并提交的 RX NPY 张量不会重算。
