# 全部训练结果（lab，ViT-B/16，10k 步）

由 `results/lab/output/*.json` 自动生成；best = 每 1000 步评估中的最高值，final = 第 10000 步。

| 数据集 | run | 输入变体 | 设置 | best | final |
|---|---|---|---|---|---|
| car | `car_dinov3` | dinov3 | — | **88.47** | 88.47 |
| car | `car_dinov3_sq` | dinov3_sq | — | **88.91** | 88.91 |
| car | `car_dinov3_sq_s1` | dinov3_sq | seed 1 | **88.75** | 88.75 |
| car | `car_gt_sq` | gt_sq | — | **89.07** | 89.07 |
| car | `car_raw` | raw | — | **88.73** | 88.73 |
| car | `car_raw_s1` | raw | seed 1 | **88.60** | 88.60 |
| cub | `cub_dinov3` | dinov3 | — | **90.20** | 90.18 |
| cub | `cub_dinov3_sq224_224` | dinov3_sq224 | img 224 | **89.30** | 89.25 |
| cub | `cub_dinov3_sq_224` | dinov3_sq | img 224 | **90.35** | 90.35 |
| cub | `cub_gt_sq_224` | gt_sq | img 224 | **89.61** | 89.45 |
| cub | `cub_rand25` | raw | random keep 0.25 | **83.69** | 83.69 |
| cub | `cub_raw` | raw | — | **90.68** | 90.66 |
| cub | `cub_raw_224` | raw | img 224 | **88.44** | 88.42 |
| cub | `cub_tok10` | raw | dinov3 keep 0.1 | **88.73** | 88.73 |
| cub | `cub_tok25` | raw | dinov3 keep 0.25 | **90.09** | 90.01 |
| cub | `cub_tok25_896` | raw | img 896, dinov3 keep 0.25 | **90.27** | 89.92 |
| cub | `cub_tok50` | raw | dinov3 keep 0.5 | **90.63** | 90.54 |
| dog | `dog_dinov3` | dinov3 | — | **90.06** | 90.06 |
| dog | `dog_dinov3_sq` | dinov3_sq | — | **90.29** | 90.15 |
| dog | `dog_dinov3_sq_s1` | dinov3_sq | seed 1 | **90.50** | 90.50 |
| dog | `dog_gt` | gt | — | **88.82** | 88.82 |
| dog | `dog_gt_sq` | gt_sq | — | **90.47** | 90.47 |
| dog | `dog_raw` | raw | — | **90.43** | 90.43 |
| dog | `dog_raw_s1` | raw | seed 1 | **90.94** | 90.94 |
| ip102 | `ip102_dinov3_sq` | dinov3_sq | — | **73.41** | 73.41 |
| ip102 | `ip102_raw` | raw | — | **73.38** | 73.38 |
| nabirds | `nabirds_dinov3_sq` | dinov3_sq | — | **91.14** | 91.14 |
| nabirds | `nabirds_dinov3_sq_s1` | dinov3_sq | seed 1 | **91.14** | 91.14 |
| nabirds | `nabirds_gt_sq` | gt_sq | — | **91.05** | 91.00 |
| nabirds | `nabirds_rand25` | raw | random keep 0.25 | **76.43** | 76.43 |
| nabirds | `nabirds_raw` | raw | — | **89.77** | 89.77 |
| nabirds | `nabirds_raw_s1` | raw | seed 1 | **89.79** | 89.74 |
| nabirds | `nabirds_tok10` | raw | dinov3 keep 0.1 | **87.63** | 87.63 |
| nabirds | `nabirds_tok25` | raw | dinov3 keep 0.25 | **89.40** | 89.40 |
| nabirds | `nabirds_tok25_896` | raw | img 896, dinov3 keep 0.25 | **90.51** | 90.51 |

## 自动框 vs 官方 bbox 的 IoU（`results/lab/logs/crops_*.log`）

- `crops_car.log`: IoU(auto, gt) all:   mean 0.801  >=0.5 0.978
- `crops_car_sq.log`: IoU(auto, gt) all:   mean 0.560  >=0.5 0.626
- `crops_cub.log`: IoU(auto, gt) all:   mean 0.796  >=0.5 0.972
- `crops_cub_sq.log`: IoU(auto, gt) all:   mean 0.463  >=0.5 0.372
- `crops_cub_sq224.log`: IoU(auto, gt) all:   mean 0.391  >=0.5 0.202
- `crops_dog.log`: IoU(auto, gt) all:   mean 0.767  >=0.5 0.896
- `crops_dog_sq.log`: IoU(auto, gt) all:   mean 0.561  >=0.5 0.581
- `crops_nabirds.log`: IoU(auto, gt) all:   mean 0.789  >=0.5 0.962
- `crops_nabirds_sq.log`: IoU(auto, gt) all:   mean 0.454  >=0.5 0.376

## RTX 3060（Windows）上的 run（日志留在那台机器，数字来自其日志）

| run | best |
|---|---|
| `cub_gt`（紧框） | 88.14 |
| `cub_dinov3_sq` | 90.58 |
| `cub_gt_sq` | 90.87 |
| `cub_raw_s1` | 90.61 |
| `cub_dinov3_sq_s1` | 90.82 |
| `speedtest`（300 步） | 66.88 |
