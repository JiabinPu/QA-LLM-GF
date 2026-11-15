# ST-Tensor 和 L1 求解模块

本模块将 ST-Tensor 与二阶 L1 滤波封装为独立 Python 接口。v1.0.2 将 `api` 和 `_kernels` 作为编译后的二进制扩展分发；独立求解仅依赖 PyTorch，不需要 GPT-2 或 LAI/FPAR 固定权重。

## 二进制安装与合成小样

本版要求 **CPython 3.12**，提供 macOS Apple Silicon arm64、Linux x86_64 和 Windows x86_64 三类二进制目标。Intel Mac、其他 Python 版本和架构不在此支持范围内。Linux 包在 Ubuntu 22.04 x86_64 / glibc 2.35、CPython 3.12 上完成构建及 14 组 CPU 对照；其他 Linux 发行版尚未验证，本包不作 manylinux 兼容性承诺。所有平台都需要兼容的 PyTorch。

完整运行包自带三平台的 native 子目录；解压后由加载器选择本机匹配的二进制，可以原地运行，无需再单独安装求解模块：

```sh
python example_st_tensor_l1.py --device cpu
```

只使用独立求解模块时，从同一 Release 下载匹配的 CPython 3.12 wheel，安装到已有 PyTorch 的环境中：

```sh
python -m pip install --no-deps /path/to/downloaded.whl
```

`api` 和 `_kernels` 只以编译扩展提供，不附带其 Python 源实现，也不支持从该公开包构建源码版本。加载器遇到不支持的平台、Python 版本或缺少匹配二进制时会清楚报错，不会悄悄调用 Python 源码或联网下载替代实现。

示例处理一个固定合成 core，形状为 `[1,64,46,5]`、ST 预算为 3，并输出实际迭代记录。它用于检查接口执行，不是实际 2013 年像元、完整生产或速度基准。

## 三种调用方式

### 单独调用数值内核

```python
from st_tensor_l1 import (
    SolverConfig,
    adaptive_sttensor,
    adaptive_l1_trend_filter,
)
```

这三个符号继续由 `qa_llm_v3.solver` 兼容导出，原调用参数不变。数值内核接收已归一化、必要时已编码的张量；不会读取模型权重或生成求解预算。

### 联合处理归一化输入

```python
from st_tensor_l1 import production_config, refine_core

result = refine_core(
    observed=observed,
    observation_weight=observation_weight,
    prior=prior,
    prior_strength=prior_strength,
    anchor_value=anchor_value,
    anchor_mask=anchor_mask,
    difficulty=difficulty,
    l1_lambda=l1_lambda,
    st_budget=st_budget,
    phenology_reference=phenology_reference,
    config=production_config(),
)

candidate = result.values
st_candidate = result.st_values
counts = result.statistics()
```

这是无梯度推理接口，处理单变量 `[N,P,T,Y]` 张量。通常 `P=64`、`T=46`，`Y` 为窗口年份数；输入数值归一化到 `[0,1]`，LAI 物理值除以 7，FPAR 保持分数尺度。`anchor_mask` 为同形状布尔张量；`prior_strength`、`difficulty`、`l1_lambda` 和整数 `st_budget` 各包含 `N` 个值。观测权重非负，所有输入位于同一设备，浮点张量采用同一 dtype，推荐沿用 FP32。

残差模式下必须提供同形状、同尺度的 `phenology_reference`。接口依次执行物候残差编码、ST、跨年 L1 和解码；结果仍为归一化的细化候选，由调用者还原物理单位。LAI 和 FPAR 分别调用与归集统计。

### 联合处理已经编码的 core

```python
from st_tensor_l1 import refine_encoded_core
```

`refine_encoded_core` 的观测、先验和锚点已经位于数值求解域，参数不包含物候参考。它只执行 ST、年份优先的序列展开、曲率保护、L1 和维度还原，返回同一编码域的结果，不重复残差编码、解码或物理值裁剪。生产流程采用这个入口，以保留原有编码及后处理的运算顺序。

独立调用默认检查输入。原生产流程已经完成张量构造，内部传入 `validate=False`，避免重复扫描与设备同步；没有增加数据转换或重新整理布局。

L1 输入为 `[N,P,Y*T]`，按年份顺序串接，不能替换为逐年独立滤波。`result.st_diagnostics`、`result.l1_diagnostics` 保留预算和实际迭代张量。

## 参数与迭代统计

`production_config()` 返回与现有固定模型匹配的配置，其中最大 ST 预算为 24、训练迭代参数为 2，且物候残差模式开启。原 `SolverConfig()` 的内核默认值仍为 40、3 和 `False`；接入现有生产时应传入原模型保存的配置。

L1 预算依据分配的 ST 预算计算，保留以下对应关系：

| ST 分配预算 | 0 | 3 | 5 | 8 | 13 | 24 |
| --- | --- | --- | --- | --- | --- | --- |
| L1 分配预算 | 0 | 1 | 2 | 2 | 3 | 5 |

ST 提前收敛不会改用实际 ST 次数重新计算 L1 预算。实际启动由诊断中的 `iterations > 0` 判断，不能以正预算数量或 QA mode 7 替代。`statistics()` 会将计数同步到 CPU，汇总预算、实际启动数和迭代分布；模块没有目标年份、植被域或控制器选择信息，因此不推断 `ValidCoreYears` 或模型路由比例。

## CPU 与可选 GPU

```sh
python example_st_tensor_l1.py --device auto
python example_st_tensor_l1.py --device cuda
PYTORCH_ENABLE_MPS_FALLBACK=1 python example_st_tensor_l1.py --device mps
```

示例的自动模式按可用性选择 CUDA、Apple MPS 或 CPU。库使用输入张量所在的设备，不自动转移设备、转换精度、整理内存布局或修改全局线程设置。MPS 的 SVD 仍可能使用 PyTorch 原生 CPU fallback；需要时在导入 PyTorch 前显式设置上面的启动环境。CUDA 接口可供具备相应硬件的用户运行，当前本机验证不包含 CUDA。

CPU/GPU 多工作进程的调度仍由上层运行时负责。封装前后应在同一后端比较数值结果，不能据此假定 CPU 与 GPU 逐值相同。

## 与完整重建流程的衔接

模块输出是数值细化候选。QA 融合、物候参考生成、GPT-2 预算、结构性缺口筛选、core 选择、目标年份写回、最终观测投影、LAI 峰值保护及 H5 编码均保留在原流程中。

无梯度推理通过联合编码接口调用，启用梯度的训练路径保留原数值调用。整个窗口预算全零时，原流程直接使用先验的旁路也保持不变，避免额外残差编码与解码的 FP32 往返。

v1.0.2 的生产流程不计算代码、模型、输入或环境的运行指纹，也不据此阻拦生产或续跑。tile、年份、patch 标识、几何和数据结构仍需匹配；文件 SHA、patch 内容校验和最终产品完整性核验仍保留。固定模型权重与示例输入保持原样，完整运行包随同一版本的 Release 提供。

本次二进制分发不能收回旧版本或 Git 历史中已公开的源码，编译也不等于保证无法查看或逆向实现。

## v1.0.2 本版实测

本机 64 项 CPU/MPS 单元测试全部通过；两个后端分别完成 14 组组件与训练梯度对照，每个后端比较 119 个张量，均逐值一致。GitHub 实际构建的 Mac wheel 另在本机通过 MPS 组件对照及 CPU/MPS 正预算窗口核对。Mac arm64、Ubuntu 22.04 x86_64 和 Windows x86_64 三个构建环境各通过 14 组同环境 CPU 对照；CUDA 没有实测。

四个固定模型小样的六层 DN 和真实迭代诊断保持同后端一致；强制正预算的 2013 窗口中间值及迭代记录也一致。真实单 patch CPU 集成完成 46 个 H5 的逐层回读，DN 与参考一致；再次运行复用缓存，新增推理数为 0。新生产记录无运行指纹。完整范围见 `binary_module_verification.json`，不表示跨平台 DN 必然相同或已经验证全量生产。

## 验证范围

下述记录对应 v1.0.1 源码封装版，不是 v1.0.2 二进制版本的新实测结果；本版二进制验证应按其发布记录中的平台、Python 和设备范围解读。历史记录不证明三平台二进制或 CUDA 都已经完成实测。

v1.0.1 的 54 项 CPU/MPS 单元测试通过。CPU 固定小样仍与随包参考逐值一致；在当前 MPS 环境中，v1.0.0 与 v1.0.1 的四个固定小样及诊断也逐值一致。另外，一个显式强制正预算的真实 2013 窗口在 CPU 和 MPS 上均与旧版保持六层 DN、中间预测和迭代记录一致。该强制检查不表示模型自然分配了相同预算。

当前系统与历史 MPS 参考的生成环境不同，两版代码相对该历史参考均出现相同的少量差异；原参考没有替换。完整记录见 `solver_encapsulation_verification.json`。
