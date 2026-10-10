![Laya Core ML 在本机玩贪吃蛇](https://raw.githubusercontent.com/mizorewww/laya-coreml/main/docs/assets/snake-demo.gif)

# Laya-CoreML

**在 Apple Silicon 上运行开放权重的决策模型：Core ML、Neural Engine、零输出 token。**

[PyPI](https://pypi.org/project/laya-coreml/) · [Hugging Face](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane) · [English](https://github.com/mizorewww/laya-coreml)

上面的贪吃蛇由真实 Core ML 推理驱动，显示方向概率、分数、蛇长、延迟和安全层接管次数。
GIF 保持录制时的 **1× 速度**。代码提供确定性的路径特征和可见的循环安全层，模型负责给出概率。

完整游戏循环在三个种子、各 600 步的不限速测试中达到 **49.1–50.0 步/秒**，零死亡、两次安全层干预。
包含终端内容生成和序列化，排除终端窗口自身绘制；[定时模式限制与完整数据](docs/SNAKE_BENCHMARKS.md) 均已公开。

**M3 Max 上，ANE FP16 单个短问题 P50 4.98 ms / P95 5.31 ms。**
同场测试中，每次决策的整机能耗比编译后的 MLX 改善 **2.78 倍**。
单独验证的 8-bit 版本为 4.88 ms、3.19 倍能效改善。**十倍目标尚未达到。**
这些数字是单问题 API 的结果，不能当成完整贪吃蛇的每帧耗时。

## 安装后直接演示

需要 Apple Silicon、macOS 15+、Python 3.11–3.13。

```bash
pip install 'laya-coreml[demo]'
hf download aac6fef/laya-multilingual-coreml-ane --local-dir models/snake
laya-coreml-snake --model ./models/snake
```

下载一次后即可离线运行。推理不依赖 PyTorch、Transformers 或 MLX。
终端至少需要 104 列 × 35 行。空格暂停，上下箭头调速，R 重开，Q 退出。
首次 Core ML 加载和编译可能需要几十秒；默认 12 步/秒便于看清概率变化。

[操作、录制与视频导出](docs/SNAKE_DEMO.md) · [稳定速度测试](docs/SNAKE_BENCHMARKS.md) · [发布素材](docs/LAUNCH.md)

## Python 调用

```python
import laya_coreml as laya

agent = laya.load("aac6fef/laya-multilingual-coreml-ane")
result = agent.predict(
    "客户要求退还重复扣除的款项。",
    {
        "refund": {
            "type": "noul",
            "instructions": "Does the customer request a refund?",
        }
    },
)
print(result["answers"]["refund"])
```

支持 `choice`、有序 `score` 和布尔 `noul`；直接返回概率，不逐 token 生成文本。
远程模型 ID 首次会下载；`local_files_only=True` 强制只使用已有缓存，也可以直接传本地目录。

跟随上游 v0.3.5，校准温度在使用前会被钳制到 `[0.5, 5.0]`：检查点自带的 `choice:11+` 桶为 0.1006，
会把 logits 锐化约 10 倍，将接近随机的答案报告成近乎确定。原始值仍可通过 `agent.temperature_raw`
和 `agent.temperature_by_options_raw` 查看，加载时会对每个被钳制的桶发出 `RuntimeWarning`。
ANE 快速版本的 **96-token 总预算包括问题、选项和状态**，超出会报错。
需要更长输入时使用 `aac6fef/laya-multilingual-coreml` 的 1024-token 通用模型。
[完整 API 和模型选择](docs/USAGE.md)。

## 性能与精度

M3 Max（40 核 GPU、128 GiB），macOS 27.2。相同短问题，MLX 开启 compile、提示词缓存和长度档位。
每组六个 20 秒区间交替执行，共 **65,598 次稳定调用**。包含分词、输入准备、同步推理和结果格式化，排除加载与预热。

| 指标 | 编译后的 MLX FP16 | ANE FP16 | ANE 8-bit |
|---|---:|---:|---:|
| P50 / P95 | 6.94 / 7.39 ms | **4.98 / 5.31 ms** | **4.88 / 5.23 ms** |
| 平均整机功率估计 | 61.39 W | 30.75 W | 27.39 W |
| 每次决策整机能耗 | 0.4288 J | 0.1540 J | 0.1344 J |
| 速度提升 | 1× | **1.39×** | **1.42×** |
| 每次决策能效改善 | 1× | **2.78×** | **3.19×** |

功耗直接读取 SMC PSTR 整机传感器，保留原始样本并拒绝异常整轮；存在传感器与后台应用误差。
速度比乘以平均功率比等于每次决策能效改善，不能把能耗比再乘一次速度。
8-bit 压缩的是权重，计算仍用 FP16；模型主体包缩小不等于同比例提速。

三个通用 FP16 模型共 **189/189** 个验证问题与原版选项一致，各完成 100 次稳定重复调用。
ANE FP16 L96 通过 **59/59**，最大校准概率偏差 0.002925；W8 同一子集偏差 0.014393，
通过既定 0.02 门槛。6-bit / 4-bit 没有通过门槛，未作为发布权重上传。
这些数字验证移植一致性，不代表任意任务的正确率。

独立导出的 ANE FP16 L1024 通过完整 **63/63**，但真正 1024-token 请求耗时约 **91.7 ms**，
没有显示出长上下文加速优势。ANE Snake 对照通过 600/600 动作一致、零死亡、零接管；
当前每步串行回答三个问题，完整决策没有显示出稳定加速，不能宣传为约 5 ms 一帧。

[完整速度/能耗报告](docs/ANE_BENCHMARKS.md) · [通用 Core ML benchmark](BENCHMARKS.md) · [原始数据](benchmarks/results)

## 六个可直接下载的模型包

| Hugging Face 模型 | 默认引擎 | 总长度 / batch |
|---|---|---|
| [laya-coreml](https://huggingface.co/aac6fef/laya-coreml) | CPU + GPU | 512 / 1 |
| [laya-multilingual-coreml](https://huggingface.co/aac6fef/laya-multilingual-coreml) | CPU + GPU | 1024 / 1 |
| [laya-typed-decisions-coreml](https://huggingface.co/aac6fef/laya-typed-decisions-coreml) | CPU + GPU | 1024 / 1 |
| [laya-multilingual-coreml-snake](https://huggingface.co/aac6fef/laya-multilingual-coreml-snake) | CPU + GPU | 64 / 3 |
| [laya-multilingual-coreml-ane](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane) | CPU + ANE | 96 / 1 |
| [laya-multilingual-coreml-ane-w8](https://huggingface.co/aac6fef/laya-multilingual-coreml-ane-w8) | CPU + ANE | 96 / 1 |

每个包包含模型卡、配置、tokenizer、校验和、来源和打包后验证。ANE 包自带所需的原始 embedding
与 action head 张量，无需原始训练仓库。普通 SDPA 导出和专门改写的 ANE 图是不同路径；
仅修改普通模型的 compute units 不会自动获得 ANE 快速路径。

## 文档

- [安装、Python 与 CLI](docs/USAGE.md)
- [贪吃蛇演示和录制](docs/SNAKE_DEMO.md)
- [发布版本和固定模型 revision](docs/RELEASE.md)
- [ANE 工程实现](docs/ANE_ENGINEERING.md)
- [十倍目标的数学调查](docs/ANE_MATH.md)
- [转换问题与修复](docs/CONVERSION.md)

代码采用 Apache-2.0。原始 [Laya](https://github.com/NandhaKishorM/laya) 模型由 Convai Innovations
及贡献者发布；本项目基于 [laya-mlx](https://github.com/mizorewww/laya-mlx) 完成独立 Core ML 移植。
这不是 Convai Innovations 或 Apple 官方发行版，归属见 [NOTICE](NOTICE)。


## v0.4.0：以官方行为为准

本版直接迁移官方 Laya v0.4.1（`1adc59f`）主机端代码，替换此前经 MLX
选择性移植的实现。默认值、输入校验、返回结构、路由与并发行为均以官方为准。
新增批量预测、长文本扫描、完整 hooks、结构化决策、语言校准与校准保存／加载，
并迁移官方路由 CLI、HTTP 服务、MCP 本地／远程工具及评估框架。
数值和布尔选项标签、`option_order`、score legend、空问题 usage 等也恢复官方语义。

迁移时注意：`load()` 默认加载英文模型，`Router()` 默认保留两个模型。
需要保留原选择时，请显式使用 `load("multilingual")` 或 `Router(max_loaded=1)`。
Core ML 模型仍保留导出的真实上下文和选项上限；ANE 的 96-token 限制不会消失。
普通预测按该预算使用官方截断逻辑，`predict_long` 按该预算扫描长文本。

详见 [0.4.0 迁移与兼容范围](docs/MIGRATION_0_4.md)、
[官方权重／配置核对](benchmarks/results/official-v041-artifact-audit.json) 和
[实际模型验证](benchmarks/results/official-host-v040-validation.json)。
测试覆盖官方原始断言、源码与差分对照，以及 macOS 上真实 Core ML 转换和推理。
推理不依赖 PyTorch、Transformers 或 MLX；可选温度拟合使用 PyTorch。
