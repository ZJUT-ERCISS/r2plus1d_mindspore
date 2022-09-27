"""EvalLossMonitor"""
import time
import os
import stat
import numpy as np

import mindspore as ms
from mindspore import save_checkpoint
import mindspore
from mindspore.train.callback import Callback
from src.utils.check_param import Rel, Validator as validator


class EvalLossMonitor(Callback):
    """
    Monitor for .

    Args:
        lr_init (Union[float, Iterable], optional): The learning rate schedule. Default: None.
        per_print_times (int): Every how many steps to print the log information. Default: 1.

    Examples:
        >>> from mindvision.engine.callback import LossMonitor
        >>> lr = [0.01, 0.008, 0.006, 0.005, 0.002]
        >>> monitor = LossMonitor(lr_init=lr, per_print_times=100)
    """

    def __init__(self, model):
        super(EvalLossMonitor, self).__init__()
        self.model = model
    # pylint: disable=unused-argument

    def epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """
        Print training info at the end of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        callback_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / callback_params.batch_num
        print(f"Epoch time: {epoch_mseconds:5.3f} ms, "
              f"per step time: {per_step_mseconds:5.3f} ms, "
              f"avg loss: {np.mean(self.losses):5.3f}", flush=True)

    # pylint: disable=unused-argument
    def step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.step_time = time.time()

    # pylint: disable=missing-docstring
    def step_end(self, run_context):
        """
        Print training info at the end of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        callback_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        loss = callback_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], mindspore.Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, mindspore.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        self.losses.append(loss)
        cur_step_in_epoch = (callback_params.cur_step_num - 1) % callback_params.batch_num + 1

        # Boundary check.
        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("Invalid loss, terminate training.")
        metrics = []
        for k, v in self.model._metrics.items():
            try:
                metrics.append(f"{k}: {v.eval():4.4f}")
            except:
                continue
        print(f"step:[{cur_step_in_epoch:5d}/{callback_params.batch_num:5d}], "
              f"metrics:{metrics}, "
              f"loss:[{loss:5.3f}/{np.mean(self.losses):5.3f}], "
              f"time:{step_mseconds:5.3f} ms, ",
              flush=True)


class ValAccMonitor(Callback):
    """
    Monitors the train loss and the validation accuracy, after each epoch saves the
    best checkpoint file with highest validation accuracy.

    Args:
        model (ms.Model): The model to monitor.
        dataset_val (ms.dataset): The dataset that the model needs.
        num_epochs (int): The number of epochs.
        interval (int): Every how many epochs to validate and print information. Default: 1.
        eval_start_epoch (int): From which time to validate. Default: 1.
        save_best_ckpt (bool): Whether to save the checkpoint file which performs best. Default: True.
        ckpt_directory (str): The path to save checkpoint files. Default: './'.
        best_ckpt_name (str): The file name of the checkpoint file which performs best. Default: 'best.ckpt'.
        metric_name (str): The name of metric for model evaluation. Default: 'Accuracy'.
        dataset_sink_mode (bool): Whether to use the dataset sinking mode. Default: True.

    Raises:
        ValueError: If `interval` is not more than 1.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> import mindspore.dataset as ds
        >>> from mindvision.classification.models import lenet
        >>> from mindvision.classification.dataset import Mnist
        >>>
        >>> net = lenet()
        >>> opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.001, momentum=0.9)
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='mean')
        >>> model = ms.Model(net, loss,opt,metrics={"Accuracy":nn.Accuracy()})
        >>> dataset_val = Mnist("./mnist", split="test", batch_size=32, resize=32, download=True)
        >>> dataset_val = dataset_val.run()
        >>> monitor = ValAccMonitor(model, dataset_val, num_epochs=10)
    """

    def __init__(self,
                 model: ms.Model,
                 dataset_val: ms.dataset,
                 num_epochs: int,
                 interval: int = 1,
                 eval_start_epoch: int = 1,
                 save_best_ckpt: bool = True,
                 ckpt_directory: str = "./",
                 best_ckpt_name: str = "best.ckpt",
                 metric_name: str = "Accuracy",
                 dataset_sink_mode: bool = True):
        super(ValAccMonitor, self).__init__()
        self.model = model
        self.dataset_val = dataset_val
        self.num_epochs = num_epochs
        self.eval_start_epoch = eval_start_epoch
        self.save_best_ckpt = save_best_ckpt
        self.metric_name = metric_name
        self.interval = validator.check_int(interval, 1, Rel.GE, "interval")
        self.best_res = 0
        self.dataset_sink_mode = dataset_sink_mode

        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.best_ckpt_path = os.path.join(ckpt_directory, best_ckpt_name)

    def apply_eval(self):
        """Model evaluation, return validation accuracy."""
        return self.model.eval(self.dataset_val, dataset_sink_mode=self.dataset_sink_mode)[self.metric_name]

    # pylint: disable=missing-docstring
    def epoch_end(self, run_context):
        """
        After epoch, print train loss and val accuracy,
        save the best ckpt file with highest validation accuracy.

        Args:
            run_context (RunContext): Context of the process running.
        """
        callback_params = run_context.original_args()
        cur_epoch = callback_params.cur_epoch_num

        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            # Validation result
            res = self.apply_eval()
            loss = callback_params.net_outputs.asnumpy()
            print("-" * 20)
            print(f"Epoch: [{cur_epoch: 3d} / {self.num_epochs: 3d}], "
                  f"Train Loss: [{loss: 5.3f}], "
                  f"{self.metric_name}: {res: 5.3f}")

            def remove_ckpt_file(file_name):
                os.chmod(file_name, stat.S_IWRITE)
                os.remove(file_name)

            # Save the best ckpt file
            if res >= self.best_res:
                self.best_res = res
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        remove_ckpt_file(self.best_ckpt_path)
                    save_checkpoint(callback_params.train_network, self.best_ckpt_path)

    # pylint: disable=unused-argument
    def end(self, run_context):
        """
        Print the best validation accuracy after network training.

        Args:
            run_context (RunContext): Context of the process running.
        """
        print("=" * 80)
        print(f"End of validation the best {self.metric_name} is: {self.best_res: 5.3f}, "
              f"save the best ckpt file in {self.best_ckpt_path}", flush=True)


class SaveCallback(Callback):
    """SaveCallback"""

    def __init__(self, eval_model, ds_eval):
        """init"""
        super(SaveCallback, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.acc = 0

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num % 10 == 0:
            result = self.model.eval(self.ds_eval, dataset_sink_mode=False)
            if result['Accuracy'] > self.acc:
                self.acc = result['Accuracy']
                print("=" * 80)
                print("ACC:: " + str(self.acc) + "\n")
                file_name = str("ARN_ucf_CROSS") + str(self.acc) + ".ckpt"
                save_checkpoint(save_obj=cb_params.train_network,
                                ckpt_file_name=file_name)
                print("Save the maximum accuracy checkpoint, the accuracy is", self.acc)
