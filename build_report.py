"""Build the editable Chinese coursework report from audited JSON/CSV/figures.

Run with Python containing python-docx. No training or numerical result entry here.
"""
import json
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent
NAMES={'baseline':'ResNet-20','dw':'DW-ResNet-20','se':'SE-ResNet-20','dw_se':'DW+SE-ResNet-20'}
ZH=['T恤/上衣','裤子','套头衫','连衣裙','外套','凉鞋','衬衫','运动鞋','包','短靴']
REPO='https://github.com/linpianpiann777-prog/fashion-resnet20-lab'


def main():
    a=json.loads((ROOT/'results/analysis.json').read_text())
    runs=json.loads((ROOT/'results/summary.json').read_text())
    env=json.loads((ROOT/'results/environment.json').read_text())
    g=a['aggregates'];s=a['selected_run'];base=g['baseline'];dw=g['dw'];combo=g['dw_se']
    doc=Document();section=doc.sections[0]
    section.page_width=Cm(21);section.page_height=Cm(29.7)
    section.top_margin=Cm(1.9);section.bottom_margin=Cm(1.8)
    section.left_margin=Cm(2.1);section.right_margin=Cm(2.1)
    normal=doc.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(10.5)
    normal.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
    normal.paragraph_format.line_spacing=1.18;normal.paragraph_format.space_after=Pt(7)
    for name,size in [('Title',24),('Heading 1',17),('Heading 2',12)]:
        st=doc.styles[name];st.font.name='Calibri';st.font.size=Pt(size);st.font.color.rgb=RGBColor.from_string('172E29')
        st.element.rPr.rFonts.set(qn('w:eastAsia'),'黑体');st.paragraph_format.space_after=Pt(10)
    doc.styles['Title'].font.color.rgb=RGBColor(0,0,0)
    # The bundled default template may carry a blue title rule; remove paragraph borders.
    for style in doc.styles:
        for border in list(style.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
    doc.core_properties.title='ResNet20轻量化结构改进与服饰分类应用'
    doc.core_properties.subject='深度学习应用实践大作业'
    doc.core_properties.author=''
    footer=section.footer.paragraphs[0];footer.alignment=2
    run=footer.add_run();fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');run._r.addnext(fld)
    def p(text,style=None):return doc.add_paragraph(text,style)
    def h(text):doc.add_heading(text,level=2)
    def page(title):
        heading=doc.add_heading(title,level=1)
        heading.paragraph_format.page_break_before=True
    def table(headers,rows,widths=None):
        t=doc.add_table(rows=1,cols=len(headers));t.style='Table Grid';t.autofit=False
        for c,text in zip(t.rows[0].cells,headers):
            c.text=str(text)
            tcPr=c._tc.get_or_add_tcPr();shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E6EFE9');tcPr.append(shade)
            for r in c.paragraphs[0].runs:r.bold=True
        trpr=t.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
        for row in rows:
            cells=t.add_row().cells
            for c,text in zip(cells,row):c.text=str(text)
        for row in t.rows:
            trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
            for i,c in enumerate(row.cells):
                if widths:c.width=Cm(widths[i])
                for pp in c.paragraphs:
                    pp.paragraph_format.space_after=Pt(4);pp.paragraph_format.space_before=Pt(4)
                    pp.paragraph_format.line_spacing=1.05
                    for rr in pp.runs:rr.font.size=Pt(9)
        p('')
    def figure(name,caption,width=6.3):
        path=ROOT/'figures'/name
        if not path.exists():raise FileNotFoundError(path)
        pp=doc.add_paragraph();pp.alignment=1;pp.add_run().add_picture(str(path),width=Inches(width))
        pp.paragraph_format.space_after=Pt(3)
        pp.paragraph_format.keep_with_next=True
        cp=p(caption);cp.alignment=1;cp.paragraph_format.space_after=Pt(9)
        for r in cp.runs:r.font.size=Pt(9);r.font.color.rgb=RGBColor.from_string('52665C')
    def pct(x):return f'{x*100:.2f}%'
    def mean_sd(v,key):return f'{g[v][key+"_mean"]*100:.2f} ± {g[v][key+"_sd"]*100:.2f}'
    doc.add_heading('ResNet20轻量化结构改进\n与服饰分类应用',0)
    p('深度学习应用实践 · 大作业报告')
    p('姓名：________________    学号：________________\n班级：________________    完成日期：2026年11月12日')
    h('摘要')
    p(f'本项目针对低分辨率单件服饰图像自动分类，实现了经典小图像 ResNet-20，并设计深度可分离卷积、SE 通道注意力及二者组合三种修改。实验在 Fashion-MNIST 上从零训练，使用相同数据划分与优化配置，每种结构运行三个随机种子。原版测试准确率为 {mean_sd("baseline","test_accuracy")}%（均值 ± 样本标准差），达到预先设定的 {pct(env["target_baseline_accuracy"])} 项目基准。')
    p(f'深度可分离卷积将参数量从 {base["parameters"]:,} 降至 {dw["parameters"]:,}，降低 {a["parameter_reduction_dw_percent"]:.2f}%；卷积及线性层 MACs 降低 {a["mac_reduction_dw_percent"]:.2f}%。全部 {len(runs)} 次训练及最终评估累计用时 {a["runtime"]["total_seconds"]/60:.2f} 分钟，满足两小时训练限制。报告同时分析精度代价、注意力收益、实测延迟和低置信度人工复核机制。')
    h('任务完成范围')
    table(['要求','对应实现'],[
        ['经典 CNN 与基准','单通道 ResNet-20；独立测试集评估；预设准确率目标'],
        ['结构修改与对照','原版、DW、SE、DW+SE；相同训练轮数和随机种子'],
        ['场景应用','本地服饰分类界面；阈值、样本、模型切换和图片上传'],
        ['真实运行与可追溯','逐轮 CSV、逐次 JSON、测试概率、权重、数据图及核验脚本'],
        ['源码与运行说明','GitHub 仓库、PyCharm 配置及重现实验命令']],[3.7,12.9])
    p('与实验三的区别：本项目研究残差分类网络及轻量化消融，不复用 U-Net/YOLO 的分割或检测流程。分类输出是整张图片的类别概率，不包含检测框。')
    p('代码仓库：'+REPO)
    page('一 选型与实验数据')
    h('开源资料调研')
    p('调研 Fashion-MNIST 官方基准仓库、akamaster/pytorch_resnet_cifar10、CNN-for-Fashion-MNIST、shiftresnet-cifar 和 SE-Net-CIFAR 后，选用小图像 ResNet-20 作为基础。相比大输入尺寸的 VGG 或面向检测的 YOLO，该方案便于在有限训练预算下重复实验，并能围绕残差连接、空间卷积和通道注意力开展可解释的改进。源码独立编写，未使用预训练权重。')
    p('课程中的“结构创新”是对成熟组件进行适配、组合与实证分析，不将已有的深度可分离卷积或 SE 机制描述为原创算法。外部公开基准的数据处理、轮数和模型不同，本报告不将其精度视为已严格复现的同协议指标。')
    h('数据组成与划分')
    table(['分区','样本数','用途'],[['训练集',env['train_size'],'官方训练集内每类保留 5400 张，用于参数学习'],['验证集',env['validation_size'],'每类 600 张，选择最佳训练轮次'],['测试集',env['test_size'],'官方独立测试集，仅最终评估']],[2.5,2.5,11.6])
    p(f'官方数据包含 {len(ZH)} 类、单通道灰度图，尺寸为 28×28。固定划分种子 {env["split_seed"]}，分层抽取验证样本；所有结构和训练种子使用完全相同的训练与验证索引，索引保存在 results/split_indices.npz。测试集不参与优化、早停、学习率调整或训练轮次选择。')
    p('类别包括：'+'、'.join(ZH)+'。四个压缩数据文件先核对官方 MD5，再读取 IDX 格式；完整性检查失败时拒绝运行。数据下载地址和校验值随代码提供。')
    h('预处理与场景约束')
    p('输入像素按 x = (uint8 / 255 − 0.5) / 0.5 归一化；训练时以 0.5 概率进行水平翻转，验证和测试不增强。没有从测试集估计均值、方差或其他预处理参数。GPU 直接保存小尺寸数据张量，避免 Windows 多进程加载开销。')
    p('应用定位为服饰入库辅助分拣或教学演示。输入应是居中、单件且背景较简单的服饰图；复杂商品照片、多人穿着图与该数据分布不同，不能依据本实验精度承诺真实业务表现。')
    page('二 网络结构与改进方法')
    figure('architecture.png','图1 网络总体结构及残差分支',6.5)
    table(['部分','输出形状','设计'],[['输入与 stem','16×28×28','单通道 3×3 卷积，BN，ReLU'],['stage 1','16×28×28','3 个基本残差块'],['stage 2','32×14×14','3 个块，首块步长为 2'],['stage 3','64×7×7','3 个块，首块步长为 2'],['分类头','10','全局平均池化与全连接层']],[3,3.6,10])
    h('基础残差连接')
    p('每个块计算 y = ReLU(F(x) + S(x))，F 为两次卷积与 BN 组成的残差分支。尺寸相同时 S 为恒等映射；下采样采用步长切片与通道零填充，即无可训练参数的 option-A shortcut。原版计数为 stem 一层、残差分支十八层卷积及一层分类器，合计二十个带权重层。将输入通道由三改为一，保持残差阶段的基本布局。')
    h('深度可分离卷积与计算量')
    p('DW 版将残差分支每个 3×3 标准卷积替换为逐通道 3×3 卷积与逐点 1×1 卷积；stem 不替换，两者之间不额外插入激活或 BN，从而尽量只改变卷积因子分解。标准卷积参数为 K²CinCout，替换后为 K²Cin + CinCout。相同输出空间尺寸时，MACs 比值为 1/Cout + 1/K²。DW 版保留“ResNet-20”骨架名称，但实际卷积算子数增加。')
    h('SE 通道重标定与组合消融')
    p('SE 在每个块第二个 BN 之后、与 shortcut 相加之前加入通道权重。全局平均池化得到 z，权重 s = sigmoid(W₂ ReLU(W₁z))，输出为 s⊙F(x)。缩减比为 8，隐藏宽度至少为 1。通过原版→SE、DW→DW+SE 两组比较，分别观察注意力在完整卷积和轻量卷积上的效果。')
    p('参数量包含卷积、线性层、BN 和 SE 的可训练参数。计算量统计单张 1×28×28 输入的 Conv/Linear MACs，含 SE 内两层卷积；不含 BN、激活、池化和逐元素乘加。FLOPs 采用 2×MACs 约定，不应称为完整运行时操作数。')
    page('三 训练配置与学习过程')
    table(['配置','实测使用值'],[['运行环境',f'Python {env["python"]}；PyTorch {env["torch"]}'],['训练设备',env['gpu']],['优化器',f'AdamW；初始学习率 {env["lr"]}；weight decay {env["weight_decay"]}'],['调度器','CosineAnnealingLR；最低学习率 0.00002'],['训练预算',f'每次 {env["args"]["epochs"]} 轮；batch size {env["args"]["batch_size"]}'],['重复实验','种子 '+', '.join(map(str,env['args']['seeds']))+'；全部结构使用同一组种子'],['精度与选择','FP32；验证准确率最高的权重；并列取更早轮次']],[3.4,13.2])
    figure('learning_curves.png','图2 训练损失与验证准确率；线为三个种子的均值，阴影为样本标准差',6.5)
    p('训练损失根据当轮增强后的样本计算，验证指标来自固定无增强样本；因此两者不是完全相同条件的直接比较。学习率逐渐下降有利于后期稳定收敛。中间验证曲线的波动可能受优化噪声和 BN 运行统计影响，不能据单个波动点判断结构优劣。')
    p('每轮保存历史记录，按验证集挑选 checkpoint 后，对官方测试集进行一次最终预测。每个结构的界面默认权重也只按验证准确率从三个种子中选择。所有种子的测试结果均完整保留，不仅报告最好一次。')
    page('四 精度与结构消融结果')
    table(['结构','准确率 %\n均值 ± SD','Macro-F1 %\n均值 ± SD','参数量','MACs / 图'],[
        [NAMES[v],mean_sd(v,'test_accuracy'),mean_sd(v,'macro_f1'),f'{g[v]["parameters"]:,}',f'{g[v]["macs"]:,}'] for v in NAMES],[3.5,3.2,3.2,2.5,4.2])
    figure('tradeoffs.png','图3 参数量、计算量与测试准确率；准确率误差线为三个训练种子的样本标准差',6.5)
    p(f'原版平均准确率 {pct(base["test_accuracy_mean"])}，高于预设 {pct(env["target_baseline_accuracy"])} 基准。DW 版平均准确率为 {pct(dw["test_accuracy_mean"])}，相对原版变化 {(dw["test_accuracy_mean"]-base["test_accuracy_mean"])*100:+.2f} 个百分点，同时参数减少 {a["parameter_reduction_dw_percent"]:.2f}%，MACs 减少 {a["mac_reduction_dw_percent"]:.2f}%。这体现了模型容量与算术开销之间的权衡。')
    p(f'SE 相对原版平均准确率变化 {a["se_delta_accuracy_pp"]:+.2f} 个百分点；DW+SE 相对 DW 变化 {a["dw_se_delta_accuracy_pp"]:+.2f} 个百分点。注意力增加的数据依赖通道权重不保证在所有骨架上带来收益，应分别考察完整卷积和轻量卷积。')
    p('上述标准差衡量随机初始化、随机批次与增强带来的训练波动，不是准确率的置信区间。三个种子共享同一个测试集，本实验未进行独立数据集重复，也未据此声称差异具有统计显著性。')
    page('五 运行成本与逐次实验记录')
    table(['结构','种子','最佳轮次','测试正确数 / 10000','训练秒数'],[
        [NAMES[r['variant']],r['seed'],r['best_epoch'],r['test_correct'],f'{r["train_seconds"]:.2f}'] for r in runs],[4.4,1.6,2.2,4.8,3.6])
    figure('latency_accuracy.png','图4 单张 GPU 延迟与精度；点为三个种子的均值，误差线为样本标准差',6.0)
    p(f'完整训练与最终评估总计 {a["runtime"]["total_seconds"]:.2f} 秒（{a["runtime"]["total_seconds"]/60:.2f} 分钟），不包含前期环境检查、下载、代码开发和文档制作。每次训练时间包含逐轮验证与最佳权重保存；总耗时还包含最终测试与延迟测量。')
    p('延迟在相同 GPU 上以 batch=1、FP32、20 次预热和 100 次正式测量获取，每次测量进行 CUDA 同步，报告单次运行中位数，再跨种子汇总。此计时不含图片解码、网络传输或界面渲染。轻量模型算子数量更多，启动和内存访问开销可能占主导，因此 MACs 降低不保证 GPU 实际延迟下降。')
    page('六 类别表现与错误分析')
    figure('confusion_matrix.png',f'图5 验证集选出的 {NAMES[s["variant"]]} seed {s["seed"]} 的测试混淆矩阵；每类 1000 张',6.1)
    worst=min(range(10),key=lambda i:s['per_class_f1'][i])
    p(f'用于详细分析的模型按验证准确率选取，为 {NAMES[s["variant"]]}，训练种子 {s["seed"]}，最佳轮次 {s["best_epoch"]}。其测试准确率为 {pct(s["test_accuracy"])}、Macro-F1 为 {pct(s["macro_f1"])}。图中行是真实类别，列是预测类别。')
    p(f'该次模型中，F1 最低的类别是“{ZH[worst]}”，精确率 {pct(s["per_class_precision"][worst])}、召回率 {pct(s["per_class_recall"][worst])}、F1 {pct(s["per_class_f1"][worst])}。相近衣物轮廓在低分辨率灰度输入中缺少颜色、材质和细节线索，容易混淆；这是一种与图像信息限制相符的解释，而不是对每个误判的因果证明。')
    p('精确率 P = TP/(TP+FP)，召回率 R = TP/(TP+FN)，F1 = 2PR/(P+R)，Macro-F1 为各类别 F1 的算术平均。混淆矩阵与全部预测概率随仓库保存，可逐样本复核。')
    page('七 误判样例与人工复核策略')
    figure('error_examples.png','图6 按原测试集顺序选取前八个误判，不按置信度或视觉效果筛选',6.0)
    figure('threshold_sweep.png','图7 固定模型的测试集阈值扫描，仅作描述性分析，不用于挑选训练权重',6.3)
    t=next(r for r in a['threshold_sweep'] if r['threshold']==.8)
    p(f'界面默认阈值为 0.80：当最大 Softmax 概率小于阈值时进入人工复核，不改变原始类别预测。对图5对应的模型，该阈值在测试集自动接受 {t["accepted"]} 张、转人工 {t["rejected"]} 张，覆盖率 {pct(t["coverage"])}，已接受样本准确率 {pct(t["accepted_accuracy"])}。这些数字仅对应此模型和测试集，不可直接外推到商品照片。')
    p('提高阈值通常会降低自动覆盖率，但不保证已接受样本准确率严格单调增加。Softmax 未进行概率校准，错误预测也可能有很高置信度。默认阈值是演示设置，没有用测试集搜索最优阈值；真实部署应在独立业务验证集上按人工成本和误判代价选择阈值。')
    page('八 可视化界面与场景应用')
    if (ROOT/'figures/ui_screenshot.png').exists():figure('ui_screenshot.png','图8 本机实际运行的服饰识别界面',5.3)
    h('使用流程')
    p('在 PyCharm 配置解释器后运行 app.py，浏览器打开本地服务。左侧切换模型、浏览测试样本或上传图片，支持上一张、下一张、编号跳转和按类随机；右侧显示类别、置信度、Top-5 概率和模型复杂度。')
    p('滑块实时更新“自动接受／待人工复核”状态，不重新训练。各模型权重按验证集选取，图片仅发送到本机 127.0.0.1，使用 CPU 推理。')
    h('场景适配与边界')
    p('上传图片经灰度转换、等比例缩放、补黑边和归一化，界面显示实际模型输入。白底图可尝试反色，但这不等于领域适配。系统仅识别十类单件服饰，不支持目标定位、多目标检测或品牌识别。')
    h('功能验证')
    p('接口测试已通过四模型预测一致性、概率归一化、阈值边界、样本切换、类别筛选、图片上传和错误输入拦截检查。浏览器中另行验证交互与布局，截图来自实际运行页面。')
    page('九 复现步骤与总结')
    h('代码仓库与关键文件')
    p(REPO)
    table(['文件或目录','用途'],[['models.py / data_utils.py','结构实现、复杂度统计、校验与分层数据划分'],['train.py / results/','训练入口、权重、逐轮日志、指标与逐样本预测'],['app.py / ui.html','PyCharm 可启动的本地交互界面'],['analyze.py / figures/','从真实结果生成分析指标和数据图'],['build_report.py / report/','自动引用实验数据生成可编辑报告'],['README.md / check_project.py / test_app.py','安装与运行说明、结构测试和界面接口验证']],[6.8,9.8])
    h('复现命令')
    p('python download_data.py\npython check_project.py\npython train.py --epochs 20 --seeds 42 43 44 --output results_reproduce\npython app.py\npython test_app.py')
    p('训练输出目录默认 results，已有完成结果时程序拒绝覆盖；复现时使用新目录保护原始证据。analyze.py 与 build_report.py 默认读取交付的 results。完整环境说明、GPU 安装选项、图表再生成方式和目录切换方法见 README。相同随机种子有助于复现，但不同 PyTorch、CUDA 和硬件可能产生数值差异。')
    h('结论与后续工作')
    p(f'本实验完成了经典残差网络从零训练、结构消融与本地交互应用。原版达到项目基准；DW 以 {a["parameter_reduction_dw_percent"]:.2f}% 的参数缩减换取容量变化，注意力的实际收益依赖骨架。资源选型应同时参考准确率、参数、MACs 与实测延迟。后续可引入真实商品数据进行独立验证、概率校准和分布外检测，并在目标 CPU 或移动设备上重新测量性能。')
    h('参考文献与项目')
    for text in [
        '[1] He K. et al. Deep Residual Learning for Image Recognition. CVPR 2016. https://arxiv.org/abs/1512.03385',
        '[2] Howard A. G. et al. MobileNets. 2017. https://arxiv.org/abs/1704.04861',
        '[3] Hu J. et al. Squeeze-and-Excitation Networks. CVPR 2018. https://arxiv.org/abs/1709.01507',
        '[4] Xiao H. et al. Fashion-MNIST. 2017. https://arxiv.org/abs/1708.07747',
        '[5] Zalando Research. Fashion-MNIST 官方仓库. https://github.com/zalandoresearch/fashion-mnist',
        '[6] akamaster. pytorch_resnet_cifar10. https://github.com/akamaster/pytorch_resnet_cifar10']:
        pp=p(text);pp.paragraph_format.space_after=Pt(3)
        for rr in pp.runs:rr.font.size=Pt(8.5)
    from report_extensions import append_extensions
    append_extensions(doc,page,p,h,figure,table)
    output=ROOT/'report';output.mkdir(exist_ok=True)
    dest=output/'CNN大作业报告.docx';doc.save(dest)
    print(dest)


if __name__=='__main__':main()
