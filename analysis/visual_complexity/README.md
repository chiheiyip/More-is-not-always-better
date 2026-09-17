# 六场景视觉复杂度计算

本目录归档六张 VR 全景图的 FC、DL 计算、GPU 数值核验及 Excel 制表代码。
只保存源代码与依赖清单，不包含场景图片、模型权重、参与者数据、计算结果或含输出的 Notebook。

## 目录与运行环境

在本目录下创建 `input`、`output`、`code/models`；这些本地目录均被 Git 忽略。
原始恢复的 Colab 代码以 `.ipy` 保存，包含 Colab 安装与挂载命令，仅供追溯；
正式入口为 `code/source/run_panorama_local.py`。

已验证的执行环境为 Windows、Python 3.12、NVIDIA RTX 4060 Laptop GPU。
`code/requirements-lock.txt` 记录本次计算环境，其他平台不保证可原样安装。
从本目录执行：

```powershell
python -m venv code/.venv
& ./code/.venv/Scripts/python.exe -m pip install --no-deps -r code/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
New-Item -ItemType Directory -Force input, output, code/models
Invoke-WebRequest -Uri 'https://media.githubusercontent.com/media/fusionlove/image-complexity/master/trained_model_inception_v3.h5' -OutFile 'code/models/trained_model_inception_v3.h5'
```

使用 `--no-deps` 是因为 `visual-clutter` 的旧依赖声明与恢复 Notebook 中指定的
NumPy/OpenCV 版本冲突；锁定文件已列出运行所需依赖。不要将此环境混入主项目环境。
模型文件必须是真实 HDF5 权重，不可用 Git LFS 指针文件替代。

## 输入与计算

将六张原始 2:1 RGB 全景图直接放入 `input`，支持 JPG/JPEG/PNG/WEBP。
文件名须包含 `C0 W15`、`C1 W15 LA` 等条件标识，允许前置序号。
六条件必须恰好为 C0/C1 × W15/W45/W75，C1 均使用 LA。

```powershell
# 完整计算，默认使用 CUDA GPU 执行 DL
./code/run.ps1

# 仅试算 C0 W45 六个面，生成 TensorFlow CPU 对照
./code/run.ps1 --backend tensorflow --smoke

# 用刚生成的 CPU results.json 核验同一模型在 GPU 上的数值
& ./code/.venv/Scripts/python.exe code/source/verify_gpu.py --reference output/<cpu-smoke-run>/results.json
```

`--input`、`--model`、`--output` 可以覆盖默认路径。
输入图片在 CPU/GPU 对照期间必须保持不变。核验脚本使用绝对差 1e-4 作为阈值，
记录实际逐面差异，失败时返回错误。

计算遵循恢复代码的定义：

- `py360convert.e2c` 将全景图转换为六个 512 × 512 的面，使用 bilinear 插值。
- FC 使用 `visual_clutter.Vlc.getClutter_FC()` 的原默认参数，在 CPU 上执行。
- DL 加载原 Inception 模型，Pillow 默认缩放至 299 × 299，使用 Inception V3
  `preprocess_input`，然后转换为 NCHW。GPU 使用 Keras 的 PyTorch CUDA 后端，
  float32 推理并关闭 TF32；TensorFlow 后端作为 CPU 对照。
- 各场景 FC、DL 分别为六面的算术平均。
- z 标准化只使用当前六个场景，标准差采用 `ddof=0`，
  `Score = 0.5 × z(FC) + 0.5 × z(DL)`。
- 不生成三分位等级，也不计算不参与上述指标的 SE。标准化前显式拒绝零方差，
  分母不加旧代码中的 `1e-12`，与 Excel `STDEV.P` 定义一致。

输出为 `output/run_<timestamp>` 下的逐面 CSV、场景 CSV、配对差值 CSV 和
`results.json`。计算失败不会用零分替代，已经完成的逐面记录会保留。
图像指标不等同于认知负荷或参与者实际感知评分。

## Excel 制表与核验

`build_results.mjs` 需要 Node.js 和 `@oai/artifact-tool`。
可从正常的 Node 模块搜索路径加载该库，或将 `ARTIFACT_TOOL_MODULE` 环境变量设为
已有运行时中该模块可解析的绝对路径；本仓库不分发该依赖。

```powershell
node code/source/build_results.mjs output/<run-directory>
powershell -File code/source/verify_excel.ps1 -RunDirectory output/<run-directory>
```

工作簿包含“原始指标与差值”“标准化与综合分”“逐面结果与说明”三个工作表。
均值、差值、标准化和综合分使用 Excel 公式，数值保留完整精度、显示六位小数。
制表器核对公式与 Python 结果，并渲染检查图；核验脚本另需本机 Microsoft Excel，
以只读方式打开工作簿，核对 48 个结果及临时输入变化的联动，再关闭而不保存改动。

## 算法与模型来源

- FC 实现：https://github.com/kargaranamir/visual-clutter
- DL 模型：https://github.com/fusionlove/image-complexity

原 Notebook 中未记录的历史依赖版本不能从代码恢复。锁定清单表示本次重算环境，
不是对历史环境或旧九场景结果的复现声明。
