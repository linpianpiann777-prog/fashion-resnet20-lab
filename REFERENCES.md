# 资料调研与来源

检索日期：2026-11-12。下列项目用于比较选型与核对网络原理；本仓库模型、训练循环和界面为独立编写，没有复制第三方实现代码。

| 来源 | 用途与取舍 |
|---|---|
| [Fashion-MNIST 官方仓库](https://github.com/zalandoresearch/fashion-mnist) | 官方数据、MD5、类别与公开基准入口。28×28 灰度服饰图片适合轻量实验。公开基准训练协议不同，不能直接据此声称复现其精度。 |
| [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) | 经典小图像 ResNet 的结构参考；本项目采用 3 个 stage、每个 stage 3 个残差块及 option-A shortcut，并将输入改为单通道。没有使用该仓库权重。 |
| [wzyjsha-00/CNN-for-Fashion-MNIST](https://github.com/wzyjsha-00/CNN-for-Fashion-MNIST) | 调研 Fashion-MNIST 上经典 CNN 的适用性。最终避开本机已有 VGG 与 YOLO 项目，选择 ResNet-20。 |
| [alvinwan/shiftresnet-cifar](https://github.com/alvinwan/shiftresnet-cifar) | 调研在 ResNet 中替换空间卷积的实验方向；本项目只比较普通与深度可分离卷积，不实现 Shift。 |
| [Queequeg92/SE-Net-CIFAR](https://github.com/Queequeg92/SE-Net-CIFAR) | 调研残差网络结合 SE 的方向；使用统一训练设置进行自己的消融实验。 |

## 原始论文

1. He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition. CVPR 2016. https://arxiv.org/abs/1512.03385
2. Howard A. G. et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 2017. https://arxiv.org/abs/1704.04861
3. Hu J. et al. Squeeze-and-Excitation Networks. CVPR 2018 / TPAMI extended version. https://arxiv.org/abs/1709.01507
4. Xiao H., Rasul K., Vollgraf R. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. 2017. https://arxiv.org/abs/1708.07747

本项目的“结构创新”指课程范围内的组件替换、组合及实证分析，不声称首次提出深度可分离卷积或 SE。
