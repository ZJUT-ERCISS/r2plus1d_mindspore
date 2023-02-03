# Contents
- [Contents](#contents)
  - [Description](#description)
  - [Model Architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Environment Requirements](#environment-requirements)
  - [Quick Start](#quick-start)
    - [Requirements Installation](#requirements-installation)
    - [Dataset Preparation](#dataset-preparation)
    - [Model Checkpoints](#model-checkpoints)
    - [Running](#running)
  - [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
  - [Training Process](#training-process)
  - [Evaluation Process](#evaluation-process)
  - [Performance](#performance)
      - [Evaluation Performance](#evaluation-performance)
  - [Benchmark](#benchmark)
  - [Visualization](#visualization)
  - [Citation](#citation)

## [Description](#contents)
  

The code of this warehouse is the implementation of R(2+1)D network based on the mindspore framework. If you want to read the original paper, you can click the link below: [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)

## [Model Architecture](#contents)

<div align=center>
<img src=./src/pic/r2plus1d.png> 

Figure1 3D vs (2+1)D convolution</div>

The figure above shows the difference between 3D convolution and (2+1)D convolution.

- Full 3D convolution is carried out using a filter of size t × d × d where t denotes the temporal extent and d is the spatial width and height. 
- A (2+1)D convolutional block splits the computation into a spatial 2D convolution followed by a temporal 1D convolution. The authors choose the numbers of 2D filters (Mi) so that the number of parameters in the (2+1)D block matches that of the full 3D convolutional block.


<div align=center>
<img src=./src/pic/r3d_block.png> 

Table1 network architectures</div>

Table 1 shows the R3D network of layer 18 and layer 34. From R3D models the authors obtain architectures R(2+1)d  by replacing the 3D convolutions with (2+1)D convolutions. In this repository, we choose 18 layers of structure to build network.
 
  

## [Dataset](#contents)

Dataset used: [Kinetics400](https://www.deepmind.com/open-source/kinetics)

- Description: Kinetics-400 is a commonly used dataset for benchmarks in the video field. For details, please refer to its official website [Kinetics](https://www.deepmind.com/open-source/kinetics). For the download method, please refer to the official address [ActivityNet](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics), and use the download script provided by it to download the dataset.

- Dataset size：

    |category|Number of data|
    |:---:|:---:|
    |Training set|234619|
    |Validation set|19761|

```text
The directory structure of Kinetic-400 dataset looks like:

    .
    |-kinetic-400
        |-- train
        |   |-- ___qijXy2f0_000011_000021.mp4       // video file
        |   |-- ___dTOdxzXY_000022_000032.mp4       // video file
        |    ...
        |-- test
        |   |-- __Zh0xijkrw_000042_000052.mp4       // video file
        |   |-- __zVSUyXzd8_000070_000080.mp4       // video file
        |-- val
        |   |-- __wsytoYy3Q_000055_000065.mp4       // video file
        |   |-- __vzEs2wzdQ_000026_000036.mp4       // video file
        |    ...
        |-- kinetics-400_train.csv                  // training dataset label file.
        |-- kinetics-400_test.csv                   // testing dataset label file.
        |-- kinetics-400_val.csv                    // validation dataset label file.

        ...
```
  
  

## [Environment Requirements](#contents)

To run the python scripts in the repository, you need to prepare the environment as follow:

- Python and dependencies
    - python==3.7.5
    - decord==0.6.0
    - mindspore-gpu==1.6.1
    - ml-collections==0.1.1
    - numpy==1.21.5
    - Pillow==9.0.1
    - PyYAML==6.0
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

  
## [Quick Start](#contents)

### [Requirements Installation](#contents)
Use the following commands to install dependencies:
```shell
pip install -r requirements.txt
```

### [Dataset Preparation](#contents)
R(2+1)D model uses [Kinetics400](https://www.deepmind.com/open-source/kinetics) dataset to train and validate in this repository.

### [Model Checkpoints](#contents)
The pretrain model is trained on the the kinetics400 dataset. It can be downloaded here: [r2plus1d18_kinetic400.ckpt](https://zjuteducn-my.sharepoint.com/:u:/g/personal/201906010313_zjut_edu_cn/EXT6cCmxV59Gp4U9VChcmuUB2Fmuhfg7SRkfuxGsOiyBUA?e=qJ9Wc1)

### [Running](#contents)
To train or finetune the model, you can run the following script:

```shell

cd scripts/

# run training example
bash train_standalone.sh [PROJECT_PATH] [DATA_PATH]

# run distributed training example
bash train_distribute.sh [PROJECT_PATH] [DATA_PATH]
```

To validate the model, you can run the following script:
```shell
cd scripts/

# run evaluation example
bash eval_standalone.sh [PROJECT_PATH] [DATA_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)
```text
.
│  infer.py                                     // infer script
│  README.md                                    // descriptions about R(2+1)D
│  train.py                                     // training scrip
└─src
    ├─config
    │      r2plus1d18.yaml                      // R(2+1)D parameter configuration
    ├─data
    │  │  builder.py                            // build data
    │  │  download.py                           // download dataset
    │  │  generator.py                          // generate video dataset
    │  │  images.py                             // process image
    │  │  kinetics400.py                        // kinetics400 dataset
    |  |  kinetics600.py                        // kinetics600 dataset
    │  │  meta.py                               // public API for dataset
    │  │  path.py                               // IO path
    │  │  video_dataset.py                      // video dataset
    │  │
    │  └─transforms
    │          builder.py                       // build transforms
    │          video_center_crop.py             // center crop
    │          video_normalize.py               // normalize
    │          video_random_crop.py             // random crop
    │          video_random_horizontal_flip.py  // random horizontal flip
    │          video_reorder.py                 // reorder
    │          video_rescale.py                 // rescale
    |          video_reshape.py                 // reshape
    |          video_resize.py                  // resize
    │          video_short_edge_resize.py       // short edge resize
    |          video_to_tensor.py               // to tensor
    │
    ├─example
    │      r2plus1d_kinetics400_eval.py         // eval r2plus1d model
    │      r2plus1d_kinetics400_train.py        // train r2plus1d model
    │
    ├─loss
    │      builder.py                           // build loss
    │
    ├─models
    |  |  base.py                               // Generate recognizer
    │  │  builder.py                            // build model
    │  │  r2plus1d.py                           // r2plus1d model
    │  │
    │  └─layers
    │          adaptiveavgpool3d.py             // adaptive average pooling 3D
    |          avgpool3d.py                     // average pooling 3D
    │          dropout_dense.py                 // dense head
    │          inflate_conv3d.py                // inflate conv3d block
    │          resnet3d.py                      // resnet backbone
    │          unit3d.py                        // unit3d module
    │
    ├─optim
    │      builder.py                           // build optimizer
    │
    ├─schedule
    │      builder.py                           // build learning rate shcedule
    │      lr_schedule.py                       // learning rate shcedule
    │
    └─utils
            callbacks.py                        // eval loss monitor
            check_param.py                      // check parameters
            class_factory.py                    // class register
            config.py                           // parameter configuration
            init_weight.py                      // init weight
            resized_mean.py                     // Calculate mean
            six_padding.py                      // convert padding list into tuple

```

### [Script Parameters](#contents)
Parameters for both training and evaluation can be set in r2plus1d18.yaml
- config for R(2+1)D, Kinetics400 dataset

```text
# model architecture
model_name: "r(2+1)d_18"   # 模型名

#global config
device_target: "GPU"
dataset_sink_mode: False
context:      # 训练环境
    mode: 0 #0--Graph Mode; 1--Pynative Mode
    device_target: "GPU"
    save_graphs: False
    device_id: 1

# model settings of every parts
model:
    type: R2Plus1d18
    stage_channels: [64, 128, 256, 512]
    stage_strides: [[1, 1, 1],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2]]
    num_classes: 400

# learning rate for training process
learning_rate:     # learning_rate规划，对应方法在mindvision/engine/lr_schedule中
    lr_scheduler: "cosine_annealing"
    lr: 0.012
    steps_per_epoch: 29850
    lr_gamma: 0.1
    eta_min: 0.0
    t_max: 100
    max_epoch: 100
    warmup_epochs: 4

# optimizer for training process
optimizer:      # optimizer参数
    type: 'Momentum'
    momentum: 0.9
    weight_decay: 0.0004
    loss_scale: 1.0

loss:       # loss 模块
    type: SoftmaxCrossEntropyWithLogits
    sparse: True
    reduction: "mean"

train:       # 预训练相关
    pre_trained: False
    pretrained_model: ""
    ckpt_path: "./output/"
    epochs: 100
    save_checkpoint_epochs: 5
    save_checkpoint_steps: 1875
    keep_checkpoint_max: 10
    run_distribute: False

eval:
    pretrained_model: ".vscode/ms_ckpts/r2plus1d_kinetic400_20220919.ckpt"

infer:       # 推理相关
    pretrained_model: ".vscode/ms_ckpts/r2plus1d_kinetic400_20220919.ckpt"
    batch_size: 1
    image_path: ""
    normalize: True
    output_dir: "./infer_output"
 
export:       # 模型导出为其他格式
    pretrained_model: ""
    batch_size: 256
    image_height: 224
    image_width: 224
    input_channel: 3
    file_name: "r2plus1d18"
    file_formate: "MINDIR"

data_loader:
    train:          # 验证、推理相关，参数与train基本一致
        dataset:
            type: Kinetic400
            path: "/home/publicfile/kinetics-400"
            split: 'train'
            batch_size: 8
            seq: 16
            seq_mode: "discrete"
            num_parallel_workers: 1
            shuffle: True
        map: 
            operations:
                - type: VideoRescale
                  shift: 0.0
                - type: VideoResize
                  size: [128, 171]
                - type: VideoRandomCrop
                  size: [112, 112]
                - type: VideoRandomHorizontalFlip
                  prob: 0.5
                - type: VideoReOrder
                  order: [3,0,1,2]
                - type: VideoNormalize
                  mean: [0.43216, 0.394666, 0.37645]
                  std: [0.22803, 0.22145, 0.216989]
            input_columns: ["video"]
    eval:          # 验证、推理相关，参数与train基本一致
        dataset:
            type: Kinetic400
            path: "/home/publicfile/kinetics-400"
            split: 'val'
            batch_size: 1
            seq: 64
            seq_mode: "discrete"
            num_parallel_workers: 1
            shuffle: False
        map: 
            operations:
                - type: VideoRescale
                  shift: 0.0
                - type: VideoResize
                  size: [128, 171]
                - type: VideoCenterCrop
                  size: [112, 112]
                - type: VideoReOrder
                  order: [3,0,1,2]
                - type: VideoNormalize
                  mean: [0.43216, 0.394666, 0.37645]
                  std: [0.22803, 0.22145, 0.216989]
            input_columns: ["video"]
    group_size: 1
```  
## [Training Process](#contents)

Run `scripts/train_standalone.sh` to train the model standalone. The usage of the script is:

```text
bash train_standalone.sh [PROJECT_PATH] [DATA_PATH]
```

You can view the results through the file `train_standalone.log`.

```text
epoch: 1 step: 1, loss is 5.963528633117676
epoch: 1 step: 2, loss is 6.055394649505615
epoch: 1 step: 3, loss is 6.021022319793701
epoch: 1 step: 4, loss is 5.990570068359375
epoch: 1 step: 5, loss is 6.023948669433594
epoch: 1 step: 6, loss is 6.1471266746521
epoch: 1 step: 7, loss is 5.941061973571777
epoch: 1 step: 8, loss is 5.923609733581543
...
```

## [Evaluation Process](#contents)

The evaluation dataset was [Kinetics400](https://www.deepmind.com/open-source/kinetics)

Run `scripts/eval_standalone.sh` to evaluate the model. The usage of the script is:

```text
bash scripts/eval_standalone.sh [PROJECT_PATH] [DATA_PATH] [MODEL_PATH]
```

The eval results can be viewed in `eval_result.log`.

```text
step:[    1/ 1242], metrics:[], loss:[1.766/1.766], time:5113.836 ms, 
step:[    2/ 1242], metrics:['Loss: 1.7662', 'Top_1_Accuracy: 0.6250', 'Top_5_Accuracy: 0.8750'], loss:[2.445/2.106], time:168.124 ms, 
step:[    3/ 1242], metrics:['Loss: 2.1056', 'Top_1_Accuracy: 0.5312', 'Top_5_Accuracy: 0.9062'], loss:[1.145/1.785], time:172.508 ms, 
step:[    4/ 1242], metrics:['Loss: 1.7852', 'Top_1_Accuracy: 0.5833', 'Top_5_Accuracy: 0.9167'], loss:[2.595/1.988], time:169.809 ms, 
step:[    5/ 1242], metrics:['Loss: 1.9876', 'Top_1_Accuracy: 0.5312', 'Top_5_Accuracy: 0.8906'], loss:[4.180/2.426], time:211.982 ms, 
step:[    6/ 1242], metrics:['Loss: 2.4261', 'Top_1_Accuracy: 0.5000', 'Top_5_Accuracy: 0.8250'], loss:[2.618/2.458], time:171.277 ms, 
step:[    7/ 1242], metrics:['Loss: 2.4580', 'Top_1_Accuracy: 0.4792', 'Top_5_Accuracy: 0.8021'], loss:[4.381/2.733], time:174.786 ms, 
...
```
## [Performance](#contents)

#### Evaluation Performance

- r(2+1)d for kinetic400

| Parameters          | GPU                                                       |
| -------------       |--------------------------------------  |
| Model Version       | r                                                       |
| Resource            | Nvidia 3090Ti                                             |
| uploaded Date       | 09/06/2022 (month/day/year)                               |
| MindSpore Version   | 1.6.1                                                     |
| Dataset             | kinetic400                                                    |
| Training Parameters | epoch = 30,  batch_size = 64                               |
| Optimizer           | SGD                                                       |
| Loss Function       | Max_SoftmaxCrossEntropyWithLogits                         |
| Top_1               | 1pc:57.3%                                                 |
| Top_5               | 1pc:79.6%                                                 |

## [Benchmark](#contents)

The original paper did not provide the performance of r2plus1d18 on the k400 dataset, but it can be found on the official website of pytorch.

<table>
	<tr>
	    <td>Model</td>
        <td>Dataset</td>
        <td colspan="2">Original Target</td>
        <td>Mindspore</td>
	</tr >
    <tr>
	    <td rowspan="2">R(2+1)D</td>
	    <td rowspan="2">Kinetics400</td>
        <td>Top1</td>
        <td>57.5%</td>
        <td>57.3%</td>
    </tr>
    <tr>
        <td>Top5</td>
        <td>78.8%</td>
        <td>79.6%</td>
    </tr>
</table>

## [Visualization](#contents)

The following graphics show the visualization results of model inference.
<div align=center>
<img src=./src/pic/result0.gif> 
<img src=./src/pic/result1.gif> 
</div>

## [Citation](#contents)


If you find this project useful in your research, please consider citing:

```BibTeX
@article{r2plus1d2018,
  title={A closer look at spatiotemporal convolutions for action recognition},
  author={Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, 
          Yann and Paluri, Manohar},
  year={2018},
  journal = {CVPR},
  doi={10.1109/cvpr.2018.00675},
}
```

```BibTeX
@misc{MindSpore Vision 2022, 
  title={{MindSpore Vision}:MindSpore Vision Toolbox and Benchmark}, 
  author={MindSpore Vision Contributors}, 
  howpublished = {\url{https://gitee.com/mindspore/vision}}, 
  year={2022} 
}
```