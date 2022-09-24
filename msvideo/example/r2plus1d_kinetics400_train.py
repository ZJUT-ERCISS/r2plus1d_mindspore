# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""r2plus1d training script."""

import argparse

from mindspore import nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from msvideo.utils.check_param import Validator, Rel
from msvideo.data import Kinetic400
from msvideo.schedule import warmup_cosine_annealing_lr_v1
from msvideo.data.transforms import VideoRandomCrop, VideoNormalize, VideoRandomHorizontalFlip
from msvideo.data.transforms import VideoResize, VideoRescale, VideoReOrder
from msvideo.models.r2plus1d import R2Plus1d18, R2Plus1d50


def r2plus1d_kinetics400_train(args_opt):
    """r2plus1d train"""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    if args_opt.run_distribute:
        if args_opt.device_target == "Ascend":
            init()
        else:
            init("nccl")

        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        dataset = Kinetic400(args_opt.data_url,
                             split="train",
                             seq=args_opt.seq,
                             num_parallel_workers=args_opt.num_parallel_workers,
                             shuffle=True,
                             num_shards=device_num,
                             shard_id=rank_id,
                             batch_size=args_opt.batch_size,
                             repeat_num=args_opt.repeat_num)
        ckpt_save_dir = args_opt.ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
    else:
        dataset = Kinetic400(args_opt.data_url,
                             split="train",
                             seq=args_opt.seq,
                             num_parallel_workers=args_opt.num_parallel_workers,
                             shuffle=True,
                             batch_size=args_opt.batch_size,
                             repeat_num=args_opt.repeat_num)
        ckpt_save_dir = args_opt.ckpt_save_dir

    # perpare dataset.
    transforms = [VideoRescale(shift=0.0),
                  VideoResize([128, 171]),
                  VideoRandomCrop([112, 112]),
                  VideoRandomHorizontalFlip(0.5),
                  VideoReOrder([3, 0, 1, 2]),
                  VideoNormalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])]
    dataset.transform = transforms
    dataset_train = dataset.run()
    Validator.check_int(dataset_train.get_dataset_size(), 0, Rel.GT)
    step_size = dataset_train.get_dataset_size()

    # Create model.
    if args_opt.model_name == "r2plus1d18":
        network = R2Plus1d18(num_classes=args_opt.num_classes,
                             pretrained=args_opt.pretrained)
    elif args_opt.model_name == "r2plus1d50":
        network = R2Plus1d50(num_classes=args_opt.num_classes,
                             pretrained=args_opt.pretrained)
    
    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_path)
        load_param_into_net(network, param_dict)

    # Set lr scheduler.
    learning_rate = warmup_cosine_annealing_lr_v1(lr=args_opt.learning_rate,
                                                  steps_per_epoch=step_size,
                                                  warmup_epochs=args_opt.warmup_epochs,
                                                  max_epoch=args_opt.epoch_size,
                                                  t_max=100,
                                                  eta_min=0)

    # Define optimizer.
    network_opt = nn.Momentum(network.trainable_params(),
                              learning_rate=learning_rate,
                              momentum=args_opt.momentum,
                              weight_decay=args_opt.weight_decay)

    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Define metrics.
    metrics = {'acc'}

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size,
        keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix='r2plus1d_kinetics400',
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)

    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

    # Begin to train.
    print('[Start training `{}`]'.format('r2plus1d_kinetics400'))
    print("=" * 80)
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor()],
                dataset_sink_mode=args_opt.dataset_sink_mode)
    print('[End of training `{}`]'.format('r2plus1d_kinetics400'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='r2plus1d train.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--run_distribute', type=bool, default=False, help='Distributed parallel training.')
    parser.add_argument('--data_url', type=str, default="", help='Location of data.')
    parser.add_argument('--seq', type=int, default=64, help='Number of frames of captured video.')
    parser.add_argument('--num_parallel_workers', type=int, default=8, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument('--ckpt_save_dir', type=str, default="./r2plus1d", help='Location of training outputs.')
    parser.add_argument("--model_name", type=str, default="r2plus1d18", help="Name of model.",
                        choices=["r2plus1d18", "r2plus1d50"])
    parser.add_argument('--num_classes', type=int, default=400, help='Number of classification.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Load pretrained model.')
    parser.add_argument('--pretrained_path', default=None, help='Path to pretrained model.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--warmup_epochs', type=int, default=4, help='Warmup epochs.')
    parser.add_argument('--epoch_size', type=int, default=100, help='Train epoch size.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the moving average.')
    parser.add_argument('--weight_decay', type=float, default=0.00004, help='Weight decay for the optimizer.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='Max number of checkpoint files.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='The dataset sink mode.')

    args = parser.parse_known_args()[0]
    r2plus1d_kinetics400_train(args)
