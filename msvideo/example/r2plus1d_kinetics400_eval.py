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
"""r2plus1d eval script."""

import argparse

from mindspore import nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from msvideo.data import Kinetic400
from msvideo.data.transforms import VideoCenterCrop, VideoNormalize
from msvideo.data.transforms import VideoResize, VideoRescale, VideoReOrder
from msvideo.models.r2plus1d import R2Plus1d18, R2Plus1d50
from msvideo.utils.callbacks import EvalLossMonitor


def r2plus1d_kinetics400_eval(args_opt):
    """r2plus1d eval"""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    dataset_eval = Kinetic400(args_opt.data_url,
                              split="val",
                              seq=args_opt.seq,
                              seq_mode="discrete",
                              num_parallel_workers=args_opt.num_parallel_workers,
                              shuffle=False,
                              batch_size=args_opt.batch_size,
                              repeat_num=args_opt.repeat_num)

    # perpare dataset.
    transforms = [VideoResize([128, 171]),
                  VideoRescale(shift=0.0),
                  VideoCenterCrop([112, 112]),
                  VideoReOrder([3, 0, 1, 2]),
                  VideoNormalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])]
    dataset_eval.transform = transforms
    dataset_eval = dataset_eval.run()

    # Create model.
    if args_opt.model_name == "r2plus1d18":
        network = R2Plus1d18(num_classes=args_opt.num_classes)
    elif args_opt.model_name == "r2plus1d50":
        network = R2Plus1d50(num_classes=args_opt.num_classes)

    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Load pretrained model.
    param_dict = load_checkpoint(args_opt.pretrained_model)
    load_param_into_net(network, param_dict)

    # Define eval_metrics.
    eval_metrics = {'Loss': nn.Loss(),
                    'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

    # Init the model.
    model = Model(network, loss_fn=network_loss, metrics=eval_metrics)

    print_cb = EvalLossMonitor(model)
    # Begin to eval.
    print('[Start eval `{}`]'.format('r2plus1d_kinetics400'))
    result = model.eval(dataset_eval,
                        callbacks=[print_cb],
                        dataset_sink_mode=args_opt.dataset_sink_mode)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='r2plus1d eval.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', type=str, default="", help='Location of data.')
    parser.add_argument('--seq', type=int, default=32, help='Number of frames of captured video.')
    parser.add_argument('--num_parallel_workers', type=int, default=1, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1, help='Number of repeat.')
    parser.add_argument("--model_name", type=str, default="r2plus1d18",
                        help="Name of model.", choices=["r2plus1d18", "r2plus1d50"])
    parser.add_argument('--num_classes', type=int, default=400, help='Number of classification.')
    parser.add_argument('--pretrained_model', type=str, default="", help='Location of Pretrained Model.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='The dataset sink mode.')

    args = parser.parse_known_args()[0]
    r2plus1d_kinetics400_eval(args)
