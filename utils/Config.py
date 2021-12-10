#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   config.py
@Time    :   2021/11/17 21:14:23
@Author  :   sakurakdx
@Contact :   sakurakdx@163.com
"""

from configparser import ConfigParser, ExtendedInterpolation
import os
import re
from transformers.training_args import TrainingArguments
from utils.log import logger


class Configurable:
    def __init__(self, path, extra_args=None) -> None:
        """读取config文件中的配置，并将额外的参数添加到类属性中

        Args:
            path (str): config文件路径
            extra_args (list, optional): 额外的参数. Defaults to None.
        """
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(path, encoding="utf-8")
        sections = config.sections()
        self._config = config
        self.path = path

        if extra_args:
            self.add_extra_args()

        # 加载文件中的参数
        for section in sections:
            items = config.items(section)
            self._add_attr(items)

        # for section in sections:
        #     print(f"[{section}]")
        #     for k, v in config.items(section):
        #         print(f"{k}={v}")

        self.save()

    def add_extra_args(self):
        """添加额外的参数"""
        extra_args = self.extra_args
        extra_args = dict(
            [(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])]
        )
        self._add_attr(extra_args)

    def _add_attr(self, items):
        """将参数添加到类属性中

        Args:
            items (dict): k：属性名，v：属性值
        """
        for k, v in items:
            k, v = self.get_type(k, v)
            if "dir" in k and not os.path.isdir(v):
                os.makedirs(v)
            self.__setattr__(k, v)

    def get_type(self, k, v):
        """
        设置值的类型
        """

        if v.lower() == "true":
            v = True
        elif v.lower() == "false":
            v = False
        elif v.lower() == "none":
            v = None
        elif "[" == v[0] and "]" == v[-1]:
            v = self.to_list(v)
        else:
            try:
                v = eval(v)
            except:
                v = v
        return k, v

    def to_list(self, v):
        v = v.replace("[", "")
        v = v.replace("]", "")
        v = v.split(",")
        temps = []
        for i in v:
            _, i = self.get_type(None, i)
            temps.append(i)
        v = temps
        return v

    def save(self):
        logger.info("Loaded config file from {} successfully.".format(self.path))
        self._config.write(open(self.save_dir + "/" + self.path.split("/")[-1], "w"))
        logger.info(
            "Write this config to {} successfully.".format(
                self.save_dir + "/" + self.path.split("/")[-1]
            )
        )
        for section in self._config.sections():
            for k, v in self._config.items(section):
                logger.info("{} = {}".format(k, v))
        
        del self._config


def get_trainArguments(config):
    TrainArgs = TrainingArguments(
        output_dir=config.output_dir,
        # str 保存路径，包括模型文件，checkpoint，log文件等
        overwrite_output_dir=config.overwrite_output_dir,
        # bool False 设置为true则自动覆写output_dir下的文件，如果output_dir指向checkpoint，则加载模型继续训练
        do_train=config.do_train,
        # bool False 和trainer没关系
        do_eval=config.do_eval,
        # bool None
        do_predict=config.do_predict,
        # bool False
        evaluation_strategy=config.evaluation_strategy,
        # str "no" 设置evaluate的策略，no：不验证，steps：每eval_steps验证一次，epoch：每一个epoch验证一次
        prediction_loss_only=config.prediction_loss_only,
        # bool False True：只返回损失，False：返回loss和自定义损失
        per_device_train_batch_size=config.per_device_train_batch_size,
        # int 8 trainer默认开启多gpu模式，这里设置每个gpu上的样本数量
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        # int 8
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # int 1 显存重计算技巧，若设为n，则forward n次之后才进行参数更新
        eval_accumulation_steps=config.eval_accumulation_steps,
        # int None 在将张量转移到cpu之前进行eval的次数
        # ---------------------------------------------------------------------- #
        # trainer默认使用AdamW优化器，以下是AdamW优化器的参数
        learning_rate=config.learning_rate,  # float 5e-5
        weight_decay=config.weight_decay,  # float 0 trainer默认不对layernorm和所有bias进行正则化
        adam_beta1=config.adam_beta1,  # float 0.9
        adam_beta2=config.adam_beta2,  # float 0.999
        adam_epsilon=config.adam_epsilon,  # float 1e-8
        # ---------------------------------------------------------------------- #
        max_grad_norm=config.max_grad_norm,
        # int 1 梯度裁剪，控制梯度的最大值
        num_train_epochs=config.num_train_epochs,
        # int -1 epochs
        max_steps=config.max_steps,
        # int -1 执行训练步骤的总数 和num_train_epochs冲突
        # ---------------------------------------------------------------------- #
        # 学习率调整策略默认为线性
        lr_scheduler_type=config.lr_scheduler_type,
        # str linear 学习率调整策略
        warmup_ratio=config.warmup_ratio,
        # float 0.0 linear初始会从0到设定的学习率，经过的步数为warmup_ratio * (len(data) / batch_size * epoch)
        warmup_steps=config.warmup_steps,
        # int 0 经过多少步到达初始学习率 和warmup_ratio冲突
        # ---------------------------------------------------------------------- #
        # ---------------------------------------------------------------------- #
        # 以下是log的相关参数
        log_level=config.log_level,
        # str passive 设置log的level，passive代表由程序设定
        log_level_replica=config.log_level_replica,
        # str passive 设置log副本的level
        log_on_each_node=config.log_on_each_node,
        # bool True 在分布式训练时，是否在每个节点分别用log记录
        logging_dir=config.logging_dir,
        # str output_dir/runs/**CURRENT_DATETIME_HOSTNAME** Tensorboard log的保存路径
        logging_strategy=config.logging_strategy,
        # str steps 日志策略，包括no，steps，epoch
        logging_first_step=config.logging_first_step,
        # bool False 是否记录和evaluate第一个step的log
        logging_steps=config.logging_steps,
        # int 500 当策略为steps时，保存log的步数
        logging_nan_inf_filter=config.logging_nan_inf_filter,
        # bool True 是否过滤nan和inf的日志，如果为True，则进行过滤，并使用平均损失代替
        # ---------------------------------------------------------------------- #
        save_strategy=config.save_strategy,
        # str steps 保存模型的策略 包括no，steps，epoch
        save_steps=config.save_steps,
        # int 500 当策略为steps时，保存模型的步数
        save_total_limit=config.save_total_limit,
        # int 保存模型的最大次数，例如设为4，则保存最近的4次模型
        save_on_each_node=config.save_on_each_node,
        # bool False 分布式训练时，保存每个节点的模型还是只保存主节点的模型
        no_cuda=config.no_cuda,
        # bool False 是否不使用cuda False：不使用
        seed=config.seed,
        # int 42
        fp16=config.fp16,
        # bool False 是否使用16位(混合)精确训练代替32位训练
        fp16_opt_level=config.fp16_opt_level,
        # str O1 对于f16训练，在[‘O0’, ‘O1’, ‘O2’, ‘O3’]中选择Apex AMP的优化级别
        fp16_backend=config.fp16_backend,
        # str auto 后端用于混合精度训练 在"auto","amp","apex"中选择
        fp16_full_eval=config.fp16_full_eval,
        # bool False 是否使用全16位精度计算而不是32位。这将更快并节省内存，但会损害度量值。
        local_rank=config.local_rank,
        # int -1 分布式培训过程的等级。
        # xpu_backend=config.xpu_backend,
        # tpu_num_cores=config.tpu_num_cores,
        # tpu_metrics_debug=config.tpu_metrics_debug,
        # debug = config.debug,
        dataloader_drop_last=config.dataloader_drop_last,
        # bool False 是否舍弃最后一个batch的数据（可能不满batch_size）
        eval_steps=config.eval_steps,
        # int None 若evaluate的策略为steps，则每eval_steps验证一次，如果不设置，则和log_step一样
        dataloader_num_workers=config.dataloader_num_workers,
        # int 0 num_works
        # 以下暂时用不到
        # past_index=config.past_index,
        # run_name=config.run_name,
        # disable_tqdm = config.disable_tqdm,
        # remove_unused_columns=config.remove_unused_columns,
        # label_names=config.load_best_model_at_end,
        load_best_model_at_end=config.load_best_model_at_end,
        # bool False 在训练结束之后是否加载最好的模型，evaluate不受影响
        # metric_for_best_model=config.metric_for_best_model,
        # greater_is_better=config.greater_is_better,
        # ignore_data_skip=config.ignore_data_skip,
        # sharded_ddp = config.sharded_ddp,
        # deepspeed=config.deepspeed,
        # label_smoothing_factor=config.label_smoothing_factor,
        # adafactor=config.adafactor,
        group_by_length=config.group_by_length,
        # bool False 是否进行动态的padding，也可以手动写collate fn来定义
        # length_column_name = config.length_column_name,
        report_to=config.report_to,
        # str all 要报告结果和日志的集成列表。包括azure_ml、comet_ml、mlflow、tensorboard、wandb。使用“all”来报告已安装的所有集成，使用“none”来报告未安装的集成。
        # ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        # dataloader_pin_memory=config.dataloader_pin_memory,
        # skip_memory_metrics=config.skip_memory_metrics,
        # use_legacy_prediction_loop=config.use_legacy_prediction_loop,
        # push_to_hub=config.push_to_hub,
        # resume_from_checkpoint=config.resume_from_checkpoint,
        # hub_model_id=config.hub_model_id,
        # hub_strategy = config.hub_strategy,
        # hub_token=config.hub_token,
        # gradient_checkpointing=config.gradient_checkpointing,
        # push_to_hub_model_id=config.push_to_hub_model_id,
        # push_to_hub_organization=config.push_to_hub_organization,
        # push_to_hub_token=config.push_to_hub_token,
        # mp_parameters=config.mp_parameters,
    )

    return TrainArgs
