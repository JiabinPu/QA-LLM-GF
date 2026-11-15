# QA-LLM-GF：CPU 与 GPU 公开示例

这是 v1.0.2 二进制发布包，包含冻结 LAI/FPAR 模型、校准参数和教学输入。ST-Tensor/L1 的 `api` 与 `_kernels` 仅提供编译二进制，不附带其 Python 源实现。CPU 与可选 GPU 的调用方式保留，生产无需重新训练或依赖原工程。作者代码、模型和派生数据许可仍待确定，见 `LICENSE_PENDING.md`。

## 1. 安装

使用 **CPython 3.12**。本版二进制目标为 macOS Apple Silicon arm64、Linux x86_64 和 Windows x86_64；Intel Mac、其他 Python 版本及架构不在支持范围内。Linux 包在 Ubuntu 22.04 x86_64 / glibc 2.35 上完成构建及 14 组 CPU 对照；其他 Linux 发行版尚未验证，不作 manylinux 兼容性承诺。需要安装兼容的 PyTorch。

完整运行包带三平台 native 子目录，解压后选择匹配二进制原地运行，不必另装求解模块。在本文件夹建立独立环境：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 使用 `.venv\Scripts\activate`。安装适合平台的 PyTorch；Mac 的 Apple GPU
由 PyTorch MPS 提供，不需要 CUDA。`requirements-macos-arm64.lock.txt` 记录本机版本，
不保证所有软件源都提供每个版本。保留 `models` 与 `assets/gpt2_first_block`，当前构造器
会先读取本地 GPT 资产，再加载完整冻结权重。无需 API key、模型服务器或外部代码目录。

若只使用 ST/L1，从同一 Release 下载匹配的 CPython 3.12 wheel，在已有 PyTorch 的环境中用 `python -m pip install --no-deps /path/to/downloaded.whl` 安装；接口见 `ST_TENSOR_L1.md`。公开包不提供核心源码构建。加载器遇到不支持的平台、Python 版本或缺少匹配二进制时会报错，不会调用源码或联网下载替代实现。

当前可参考的本机实测来自记录所列的 CPU/MPS 环境，CUDA 尚未本机实测。旧版源码已经公开，本次编译分发不能收回旧源码，也不承诺绝对不可逆向。

## 2. 默认有 GPU 就一起用，没有就用 CPU

配对生产的默认 `--device auto`：有 GPU 时使用 **5 个 CPU 进程＋5 个共享同一块
GPU 的进程**；无 GPU 时使用 **5 个 CPU 进程**。五个 GPU 进程不是五块显卡。
每个数值进程使用一个 CPU 线程；CPU 核数较少时自动减少实际进程数并记录，单核无 GPU 机器仍可用 CPU1；内存较小时减少进程数，五加五不是所有机器的最优值。

`--device cpu` 强制只用 CPU。`cpu_gpu` 无 GPU 时也可回退 CPU，并在
`execution_resources.json` 记录请求资源、实际资源和原因。显式 `mps/cuda` 是严格设备
请求，不可用时直接报错。`--gpu-device auto` 选择联合模式的加速器。

单项 worker 数为 0 可关闭该组，两项不能同时为 0。`auto/cpu_gpu` 无 GPU 且请求 CPU0
时，例外启用 CPU1 并记录原因；强制 CPU 要求 CPU 数大于 0，显式 MPS/CUDA 要求 GPU
数大于 0。训练和旧单变量示例仍选择单设备，配对生产的 `auto` 才协调 CPU/GPU 工作组。

## 3. 冻结模型教学区域

输入是 **h27v06** 的原生 200×200 裁块，保留 2001–2024 年、每年 46 期、三传感器、
语义 QA、StdDev 和年度 LC。后面的 Mac 四样本基准也来自这份输入。

```sh
python example_02_produce_region.py --check-only
python example_02_produce_region.py --device cpu --smoke --output runs/smoke_cpu
python example_02_produce_region.py --device mps --smoke --output runs/smoke_mps
```

显式 MPS 指令需要可用的 Apple GPU。`--smoke` 只运行 2013 年首个原生 40×40 patch。
不指定输出时，smoke 使用 `runs/production_smoke_<device>/`，例如
`production_smoke_auto`。完整教学区域的自动模式为：

```sh
python example_02_produce_region.py --device auto --cpu-workers 5 --gpu-workers 5 --threads 1
```

完整配对教学例包含 25 个 patch、24 年，每年 46 个、共 1,104 个最终 H5，位于
`<output>/Products/<year>/`，默认根目录为 `runs/production_200x200_<device>/`。
`--years 2013` 可只指定目标年，邻年输入上下文仍保留。
`--output` 选择独立结果目录。`--check-only` 不做推理、不写输出。

旧的单变量教学界面仍可使用，但请与配对结果分目录：

```sh
python example_02_produce_region.py --variable LAI --device cpu --output runs/single_variable
python example_02_produce_region.py --variable FPAR --device cpu --output runs/single_variable
```

这两条必须顺序执行。单变量使用 `cpu/mps/cuda/auto` 的一个设备，不支持 `cpu_gpu`
或 `--replay-map`，不产生配对求解器统计。它仍写 `Parts/`，两个变量完成后合并到
`<output>/<year>/`，重跑可能覆盖同名教学结果；不要混入配对生产目录。

## 4. 外部 compact 数据的完整 tile

用外部 tile 目录替换 `<tile-directory>`：

```sh
python example_03_produce_tile.py --data-root <tile-directory> --years 2001:2024 --device auto --cpu-workers 5 --gpu-workers 5 --threads 1 --output runs/native_tile --check-only
python example_03_produce_tile.py --data-root <tile-directory> --device auto --cpu-workers 5 --gpu-workers 5 --threads 1 --output runs/native_tile_smoke --smoke
```

第一条只检查格式与设置；第二条只运行 2013 年首个原生 40×40 patch。
确认要完整生产时，从第一条去掉 `--check-only`。完整 tile 输入不随包提供，布局为：

```text
<tile-directory>/
  Module00_Landcover/Masks/LC_Type_3_vegetation_mask_<tile>_<year>.h5
  Module01_Intermediate/
    CompactStacks/<tile>/<year>/CompactQA_<tile>_<year>_R01_C01.h5
    PythonDataset/token_index_python.csv
```

保留每年全部 16 个 block、2001–2024 年上下文、年度 LC 与原生索引。
2400×2400 tile 每年有 3,600 个 40×40 patch，求解核心为 8×8。
入口读取已经处理的 compact 数据，不负责从原始 MODIS/VIIRS HDF 生成 compact。
输入尺度、缺失值、传感器顺序和饱和 StdDev 规则见 `MODEL_AND_DATA_CARD.md`。
默认 tile 输出为 `runs/production_<tile>_<device>/`，smoke 则为
`runs/production_<tile>_smoke_<device>/`。设备名进入默认目录，避免 CPU/MPS/auto
教学指令意外共用对方的缓存。

## 5. 配对生产的续跑、复现映射与统计

默认 `--variable both` 区域生产和 tile 生产保留输出锁、patch 缓存与完整性核验。v1.0.2 不计算代码、模型、输入或环境的运行指纹，也不因运行指纹而拦截生产或续跑。tile、目标年份、patch 标识、几何和必要数据结构仍会检查，文件 SHA、patch 校验和年度成品封印继续保留。结构兼容不代表不同输入或设置在科学上等价，请自行保留比较所需的版本与数据记录。不要多个任务同时写一个目录。

`backend_assignments.jsonl` 在派发前持久记录 patch 使用的设备。
完成后 `execution_manifest.json` 保存各 patch 后端和六层校验和；用
`--replay-map` 在新目录按同一设备分配复现：

```sh
python example_02_produce_region.py --smoke --device auto --output runs/paired_smoke
python example_02_produce_region.py --smoke --device auto --replay-map runs/paired_smoke/execution_manifest.json --output runs/paired_smoke_replay
```

输入与目标范围需兼容，已记录的 CPU/MPS/CUDA 后端都必须可用。
续跑或 replay 不会把已分配的 GPU patch 悄悄改成 CPU；缺少必需后端会报错。
新动态调度可能改变 patch 分配，不能因此要求 CPU 与 MPS 的 DN 彼此相同。

配对输出：

| 位置 | 内容 |
|---|---|
| `Patches/<year>/` | 六层原生 patch、真实求解诊断及校验和 |
| `Products/<year>/` | 46 个最终 H5 与 `annual_complete.json` 完成封印 |
| `Products/solver_by_year_variable.csv` | 分年、分变量统计 |
| `Products/solver_total_by_variable.csv` | 各变量总计 |
| `Products/solver_budget_distribution.csv` | 预算与实际迭代数分布 |
| `Products/solver_statistics_notes.txt` | 统计单位、分母与解释 |
| `execution_resources.json` | 请求/实际资源与回退原因 |
| `product_summary.json` | 完成状态与六层组装回读校验 |

没有求解器 runtime CSV。ST/L1 启动比例来自原生核心实际迭代数，不用 QA 模式 7
像元覆盖率或候选比例冒充。比较统计前先看 notes 中的范围、分母和变量槽。
组装后会逐 patch 回读最终 H5 的六层 DN，与源 patch 比较通过才封印年度结果。

每个最终 H5 有 LAI、FPAR、两层不确定性和两层 QA 六个 uint16 数据集。
物理量和不确定性乘 0.001，先排除填充值 65535；QA 不缩放。
不确定性是校准 sigma，不是 95% 区间半宽；MATLAB 直接读行列，无需转置。

## 6. 历史验证记录的范围

下述四样本、10 patch 和组件检查对应先前源码发布版，不是 v1.0.2 二进制包的新实测结果。本版应按发布验证记录中的具体 Python、平台和设备范围使用，不能把历史 Mac CPU/MPS 结果视为 Linux、Windows 或 CUDA 二进制均已实测。


```sh
python verify_reproduction.py --device cpu --output runs/reproduction_cpu.json
python verify_reproduction.py --device mps --output runs/reproduction_mps.json
```

CPU、MPS 各自对照 4 个固定 h27v06 样本：2013 年 `(0,0)`、`(0,40)`，
2001 年和 2024 年的 `(0,0)`，均为零起点 patch 原点。六层 uint16 DN 都是
**逐值 0 差异**，真实求解器诊断也与对应设备基准一致。
基准为 `data/mac_reference/reference_cpu.npz`、`reference_mps.npz`：
使用本包 h27v06 输入、冻结权重，经保留的本机 Mac 运行时新生成的小样。
运行验证不需要原工程，也不以旧 Windows DN 为目标。

另有 5 CPU＋5 MPS 的 10 patch 联合检查，各设备执行 5 个 patch，46 个最终 H5
逐 patch 六层组装回读通过。最终 replay 的设备映射完全一致、10 个六层 patch 校验和
全部相等，3 张求解器 CSV 除运行指纹外全部数值相等。随后续跑复用全部 10 个缓存，
新增推理为 0，46 个年度 H5 校验通过。这是小范围集成检查，不是速度 benchmark，
也不是完成 200×200 全部范围或完整 tile 生产。

打包时还用外部 h23v04 的 `(2013,0,0)` 分别检查 CPU、MPS：新旧预处理每值及
stride、六层 DN、实际求解器诊断全部一致。该外部 native tile 输入不随包提供，
包内可复现基准仍是前述 4 个 h27v06 样本。

上述 h27v06 和 h23v04 完整链对照样本的求解预算均为 0。CPU/MPS 的 4 项独立
原生核心组件测试均通过，
覆盖正预算 ST=3、L1=2；MPS 的 ST 实际运行 3 轮、QR 6 次、SVD 9 次，出现预期的
Torch SVD CPU fallback 提示；L1 实际运行 2 轮、Cholesky 1 次、solve 2 次。
组件检查不能写成完整链正路由已验收。

模拟 GPU 不可用时，`auto` 的 1 patch CPU 回退及 46 个 H5 组装也已通过。
全部 40 项单元测试通过，包括 32 项 CPU 调度/stride/统计检查和上述 4 项 CPU/MPS
正预算组件检查，结果登记于 `environment_tested.json`。
CUDA 尚未实测，以上都不是速度 benchmark。运行保持 FP32、每进程单线程及原流程
stride/offset 传输，MPS 不支持操作采用 CPU fallback。

原完整 Mac 正式 H5 已删除，因此这里不是对已删全年成品的回归；外部 SI TIFF 也不能
替代六层 QA-LLM-GF 基准。本候选包没有重新训练或完整 tile 生产，软件对照不代表论文
科学精度或跨平台逐位相等。

## 7. 小样本训练与 MATLAB

训练输入来自 h23v04 八个空间位置：训练四个、验证两个、测试两个，各保留 2011–2015 年。

```sh
python example_01_train_validate.py --variable LAI --stage all
python example_01_train_validate.py --variable FPAR --stage all
python example_01_train_validate.py --variable both --stage all --smoke
```

阶段为 `prepare/train/budget/evaluate/uncertainty/all`。`prepare` 仅整理 compact 教学数据
的索引和遮蔽合同；默认两轮控制器训练、三轮预算头训练。输出在 `runs/tutorial/` 或
`runs/tutorial_smoke/`，保留 loss、物理/归一化 RMSE、验证/测试与校准表。
LAI 归一化除以 7，FPAR 除以 1。小样本展示学习链，不替代冻结生产模型或论文精度。
测试集不选模型、不拟合校准；验证集被多阶段复用，不是全新的独立 conformal split。

```matlab
pythonExe = 'python';
Example01_Train_Validate('LAI', 'all', pythonExe);
Example02_Produce_Region('both', 'smoke', pythonExe, 'runs/matlab_smoke', 'auto', 5, 5);
```

`Example03_Produce_Tile.m` 提供外部 compact tile 的 MATLAB 入口，默认先检查。
MATLAB 只启动与读取，Python 完成数值运算；读取用 `Read_Example_Product.m`，
无需 `pyenv`、MATLAB GPU 工具箱或 Mapping Toolbox。
`core` 包含算法接口、重建流程与 ST/L1 编译二进制，`config` 为教学设置，`models/assets` 为冻结资源，`data` 包含教学输入
和小型 Mac 基准，`third_party` 为第三方许可；结果在运行后生成到独立目录。
包内不附带旧生产日志、完整 tile、虚拟环境或个人绝对路径。
