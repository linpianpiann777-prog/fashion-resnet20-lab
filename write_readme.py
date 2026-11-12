"""Update README from the same audited analysis used by the report."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    a=json.loads((ROOT/'results/analysis.json').read_text());g=a['aggregates']
    names={'baseline':'ResNet-20','dw':'DW-ResNet-20','se':'SE-ResNet-20','dw_se':'DW+SE-ResNet-20'}
    rows=[]
    for v,name in names.items():
        x=g[v]
        rows.append(f'| {name} | {x["parameters"]:,} | {x["macs"]/1e6:.3f} | {x["test_accuracy_mean"]*100:.2f} ± {x["test_accuracy_sd"]*100:.2f} | {x["macro_f1_mean"]*100:.2f} | {x["latency_ms_median_mean"]:.3f} |')
    text='''# Fashion Lab：ResNet-20 轻量化改进与服饰分类

深度学习课程大作业：**真实训练、四组结构消融、三个随机种子、可调置信度的本地识别界面、自动生成中文 Word 报告**。

经典架构采用小图像 ResNet-20，与 U-Net/YOLO 检测或分割实验不同。修改为深度可分离卷积（DW）、SE 通道注意力、DW+SE 组合。只使用 Fashion-MNIST，不依赖预训练模型。

![界面](figures/ui_screenshot.png)

## 已完成的真实实验

环境：Windows 11、RTX 4060 Laptop 8GB、Python 3.12.10、PyTorch 2.11.0+cu128。
每种结构训练 20 轮，种子 42/43/44；训练/验证/测试 = 54000/6000/10000，固定分层划分。
测试准确率为三个种子的**均值 ± 样本标准差**，不是挑选最好一次。预设原版准确率目标为 90%，不是声称复现某篇论文的同协议数值。

| 结构 | 参数量 | MACs (M) | 测试准确率 (%) | Macro-F1 均值 (%) | GPU 单张延迟 (ms) |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

全部 12 次训练和最终评估总计 **{a['runtime']['total_seconds']/60:.2f} 分钟**（不含下载、开发和报告制作）。DW 参数减少 **{a['parameter_reduction_dw_percent']:.2f}%**，MACs 减少 **{a['mac_reduction_dw_percent']:.2f}%**。更少 MACs 不等于更低 GPU 延迟；小算子启动和访存也影响速度。

![结构对比](figures/tradeoffs.png)

## 本机直接在 PyCharm 运行

本机已验证的 CUDA 环境可直接复用，不必重新下载或安装大环境：

1. 在 PyCharm 中打开 `E:\\deeplearn`。
2. Settings → Project → Python Interpreter → Add Interpreter → Existing，选择 `E:\\yolo_course\\.venv\\Scripts\\python.exe`。
3. 运行配置的 Script path 选择项目内 `app.py`，Working directory 选择 `E:\\deeplearn`。
4. 右键 `app.py` → Run。浏览器访问 <http://127.0.0.1:7860>。
5. 停止：PyCharm 停止按钮或终端 Ctrl+C。

PowerShell 也可直接运行：

```powershell
cd E:\\deeplearn
& E:\\yolo_course\\.venv\\Scripts\\python.exe app.py
```

仓库附带已训练权重，无需重新训练即可运行界面；首次获取项目仍需下载约 31MB 数据集供样本浏览。

## 在其他电脑安装

推荐 Python 3.12。仅界面演示可使用 CPU，安装后运行：

```powershell
git clone https://github.com/linpianpiann777-prog/fashion-resnet20-lab.git
cd fashion-resnet20-lab
py -3.12 -m venv .venv
.venv\\Scripts\\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.venv\\Scripts\\python -m pip install numpy==2.0.2 Pillow==13.0.0 matplotlib==3.11.2 python-docx
.venv\\Scripts\\python download_data.py
.venv\\Scripts\\python app.py
```

GPU 重训练使用 PyTorch 2.11.0+cu128。官方安装源：<https://download.pytorch.org/whl/cu128/torch/>。
**Windows CUDA torch 安装包约 2.75GB**，本次任务未重新下载；本机已有安装包位于 `E:\\yolo_course\\wheels`。若需重新配置，先手动下载对应 Python 3.12 / Windows 的 whl，再安装。不要把 CPU 版本直接当作 GPU 版本使用。

本项目不用 torchvision；PyTorch、NumPy、Pillow 是训练/演示主要依赖；Matplotlib 用于图表；python-docx 用于报告。`requirements.txt` 记录相应版本。报告在专用文档环境中生成，无需为运行界面安装 Office。

## 数据下载与离线放置

```powershell
python download_data.py
```

下载脚本仅下载官方四个文件，共 30,878,645 字节，并校验 MD5。
若网络受限，可在 [官方目录](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion) 手动下载以下 `.gz` 文件，不必解压，放入 `data/FashionMNIST/raw/`：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

训练和界面不会隐式联网下载，文件缺失或校验失败会给出错误。

## 重新训练与验证

以下 `python` 指所选虚拟环境解释器。

```powershell
python check_project.py
python train.py --benchmark --output benchmark_local
python train.py --epochs 20 --seeds 42 43 44 --output results_reproduce
python test_app.py
python verify_results.py
```

`results_reproduce` 用于保护交付的 `results` 原始证据，不会自动切换界面或报告的数据来源。训练默认预算为 6900 秒，到时停止并保留中间记录；实际速度取决于硬件。CPU 训练不保证两小时内完成。训练进程使用 GPU，界面使用 CPU。

结构检查涵盖：SE/DW 模块数量、反向梯度、输出尺寸、MD5、训练验证集合不相交。接口测试覆盖四模型权重与保存测试概率一致、阈值切换、样本切换、按类随机、上传和错误输入。

## 重新生成图表与 Word 报告

```powershell
python analyze.py
python build_report.py
python write_readme.py
```

这三个脚本读取交付的 `results`，不重新训练、不填写虚构结果。若需要使用复现实验结果制作新报告，请先备份原始 `results`，再将完整的新结果目录复制为 `results`，依次运行上述命令。应用重启后会重新选择相应验证集最佳权重。

报告：[`report/CNN大作业报告.docx`](report/CNN大作业报告.docx)。封面的姓名、学号、班级留空，提交前填写。

## 界面功能与适用范围

- 四种网络切换；每种网络使用验证集选出的最佳种子权重。
- 上一张、下一张、随机样本、按类别随机、编号跳转。
- 0–100% 置信度滑块；低于阈值显示“待人工复核”，不改变模型类别预测。
- PNG/JPG 上传，灰度转换、等比例缩放及补黑边，可选择反色；显示实际 28×28 模型输入。
- 显示 Top-5 概率、真实类别、参数量及 MACs。

Softmax 置信度没有经过概率校准，不等同于正确率；模型仅适用于十类单件服饰的教学演示。复杂真实商品照片属于分布偏移，需额外数据验证。不包含检测框、多目标识别或分割。

## 证据目录

```text
models.py                  四种结构与复杂度统计
data_utils.py              官方 MD5、IDX 加载和固定分层划分
train.py                   训练、验证选权重、最终测试、GPU 延迟
app.py / ui.html           本机服务与交互界面
results/environment.json   环境、超参数、数据规模和统计口径
results/split_indices.npz  固定训练验证索引
results/*/history.csv      每轮损失、精度、学习率和耗时
results/*/best.pt          每次运行的验证集最佳权重
results/*/metrics.json     每次运行的最终指标
results/*/test_predictions.npz  全部测试概率与标签
results/*/confusion_matrix.csv 混淆矩阵
results/summary.json       12 次完整结果
results/analysis.json      汇总、结构差值、阈值统计
results/comparison.csv     可用 Excel 打开的结果表
results/runtime.json       整个实验总时间
results/app_checks.json    实际接口验证记录
results/reproducibility_checks.json 重新加载全部权重复算测试集的核验记录
figures/                   PNG、SVG 图表及来源记录
report/                    可编辑 Word 报告
REFERENCES.md              开源项目调研与论文来源
```

统计口径：参数包括所有可训练参数；MACs 仅统计 Conv/Linear（含 SE 的两个卷积），不含 BN、激活、池化及逐元素运算，FLOPs=2×MACs。延迟为 GPU batch=1，20 次预热、100 次同步测量后的中位数，再对三个训练种子求均值；不是界面端到端延迟。

随机种子固定、cuDNN deterministic 开启，跨硬件/库版本仍可能存在数值差异。所有结构使用同一训练协议，不根据测试结果调整超参数；三个种子的样本标准差不是统计显著性证明。

## 来源和许可

参见 [REFERENCES.md](REFERENCES.md)。独立实现的代码采用 MIT 许可；Fashion-MNIST 由官方项目提供，下载时遵循其许可。仓库不包含数据集、虚拟环境、安装包、访问凭据或其他课程项目文件。
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')
    print(ROOT/'README.md')

if __name__=='__main__':main()
