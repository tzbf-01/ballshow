### 基于复杂运动场景的篮球持球人身份重识别算法开发

#### 赛题背景

本赛题要求在真实的”球秀“业务场景中检索特定持球人。相较于传统的行人重识别任务，本赛题针对与特定的篮球场景，若直接使用现有的成熟ReID算法会面临三大挑战：

​	1.**相似球衣区分：** 场上队友穿着完全相同的球衣，仅靠颜色无法区分。算法必须具备细粒度特征提取能力，依赖球衣号码（常被手臂或球体遮挡）、体态、护具、发型及鞋履颜色来进行区分。

​	2.**严重遮挡与姿态多变：** 持球人是防守方的重点照顾对象，常处于多人包夹（重叠遮挡）状态；同时，运球、上篮、投篮等动作导致身体姿态发生剧烈形变，与常规站立姿态差异巨大。

​	3.**环境干扰：** 比赛数据涵盖室内木地板（强反光）、室外塑胶场（复杂背景）、夜间灯光等多种环境。持球人的移动速度通常最快，导致图像极易产生运动模糊，丢失纹理细节。

我们需要你针对相应挑战，设计高效鲁棒的算法，在不依赖连续轨迹追踪的前提下，仅凭外观特征在跨时刻的图像库中精准找回目标球员。

#### 实验结果

我们选取了更适合处理长距离依赖和保留细粒度特征的**TransReID**作为本赛题的基准模型，并在本赛题所提供的数据集上进行了测试，结果如下：

|    **评估指标**     | **baseline结果** | **赛题达标要求** |
| :-----------------: | :--------------: | :--------------: |
|       **mAP**       |    **91.1%**     |     **≥91.5%**     |
| **Rank-1 Accuracy** |    **93.6%**     |     **≥94%**     |

#### 安装与使用

##### baseline源代码链接

本赛题基准基于**TransReID**实现，官方代码链接：https://github.com/damo-cv/TransReID

##### 环境依赖

```python
Python 3.7+，PyTorch 1.7.1+
# 其余依赖
pip install timm yacs termcolor
pip install -r requirements.txt
```

##### 数据准备

请将本赛题提供的训练集与测试集解压至data/BallShow/目录，并保持如下结构

```
data/BallShow/
    ├── bounding_box_train/
    ├── bounding_box_test/
    └── query/
```

##### 预训练权重

下载在ImageNet上预训练的Transformer模型：https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth

##### 路径修改

将项目configs文件夹的配置文件的相关路径设置为自己保存的路径！！

##### 训练与测试

训练命令（推荐使用 `stride` 版本以保留更多细节）

```
python train.py --config_file configs/BallShow/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
```

测试命令

```
python test.py --config_file configs/BallShow/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"  TEST.WEIGHT 'logs/BallShow_vit_transreid_stride/transformer_120.pth'
```

#### 引用

本赛题基准**TransReID**参考自：

```
@InProceedings{He_2021_ICCV,
    author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
    title     = {TransReID: Transformer-Based Object Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15013-15022}
}
```

