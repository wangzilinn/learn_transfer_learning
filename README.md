# TS相关:

1. conda环境中使用pip下载TS,不要使用conda 会有问题
2. 以下针对TS 2.5.0

# 下载:

TS官方教程:https://www.tensorflow.org/install/gpu?hl=zh-cn#software_requirements

娘的,照着官方教程来会有错误

## [NVIDIA® GPU 驱动程序](https://www.nvidia.com/drivers)

这个下载最新的

## [CUDA® 工具包](https://developer.nvidia.com/cuda-toolkit-archive)

这个要下载11.1.0版本, 不要按照官网上说的下载11.0.0版本, 安装好之后要重启

## CUDA® 工具包附带的 [CUPTI](http://docs.nvidia.com/cuda/cupti/)

这个安装CUDA工具包之后会自动安装

## [cuDNN SDK 8.0.4](https://developer.nvidia.com/cudnn)

这个下载支持11.1的cudnn-11.2-windows-x64-v8.1.0.77版本

之后把下载好的压缩包解压后随便放到哪个位置,例如`C:\cuda`

# 配置:

环境变量PATH:

前三个时CUDA工具包的配置

最后一个是cuDNN的配置

```bash
PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin
PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\extras\CUPTI\lib64
PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include
PATH=C:\cuda\bin
```

