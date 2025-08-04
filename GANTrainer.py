from models.utils.base_cli import BaseCLI

import os
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import lazy_instance
from packaging.version import Version
from torch import Tensor
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint

import models.utils.general_steps as GS
from models.io.loss import *
from models.io.norm import Norm
from models.io.stft import STFT
from models.utils.metrics import (cal_metrics_functional, recover_scale)
from models.utils.base_cli import BaseCLI
from models.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
from models.utils.my_earlystopping import MyEarlyStopping as EarlyStopping
import dataloaders


class GANTrainModule(pl.LightningModule):
    """GAN训练模块，用于语音生成任务的对抗训练
    """
    name: str  # 用于CLI创建日志目录
    import_path: str = 'Trainer.GANTrainModule'

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        recon_loss: nn.Module,  # 重构损失模块
        stft: STFT = STFT(n_fft=256, n_hop=128, win_len=256),
        norm: Norm = Norm(mode='none'),
        
        # 损失函数权重
        adv_loss_weight: float = 1.0,
        recon_loss_weight: float = 100.0,
        fm_loss_weight: float = 10.0,  # feature matching loss
        
        # 优化器配置
        gen_optimizer: Tuple[str, Dict[str, Any]] = ("Adam", {"lr": 0.0002, "betas": (0.5, 0.999)}),
        disc_optimizer: Tuple[str, Dict[str, Any]] = ("Adam", {"lr": 0.0002, "betas": (0.5, 0.999)}),
        
        # 学习率调度器
        gen_lr_scheduler: Optional[Tuple[str, Dict[str, Any]]] = None,
        disc_lr_scheduler: Optional[Tuple[str, Dict[str, Any]]] = None,
        
        # 评估指标
        metrics: List[str] = ['SDR', 'SI_SDR', 'PESQ', 'STOI'],
        val_metric: str = 'gen_loss',
        write_examples: int = 50,
        sample_rate: int = 16000,
        
        # 训练参数
        disc_start_epoch: int = 0,  # 判别器开始训练的epoch
        gen_train_freq: int = 1,    # 生成器训练频率
        disc_train_freq: int = 1,   # 判别器训练频率
        
        # 其他参数
        compile: bool = False,
        exp_name: str = "gan_exp",
        reset: Optional[List[str]] = None,
    ):
        """
        Args:
            generator: 生成器网络
            discriminator: 判别器网络
            recon_loss: 重构损失模块 (nn.Module)
            adv_loss_weight: 对抗损失权重
            recon_loss_weight: 重构损失权重
            fm_loss_weight: 特征匹配损失权重
            disc_start_epoch: 判别器开始训练的epoch
            gen_train_freq: 生成器训练频率
            disc_train_freq: 判别器训练频率
        """
        super().__init__()
        
        # 保存所有超参数
        self.save_hyperparameters(ignore=['generator', 'discriminator', 'recon_loss'])
        
        # 网络模型
        if compile and Version(torch.__version__) >= Version('2.0.0'):
            self.generator = torch.compile(generator, dynamic=Version(torch.__version__) >= Version('2.1.0'))
            self.discriminator = torch.compile(discriminator, dynamic=Version(torch.__version__) >= Version('2.1.0'))
        else:
            self.generator = generator
            self.discriminator = discriminator
        
        # 损失函数模块
        self.recon_loss = recon_loss
            
        # STFT和标准化
        self.stft = stft
        self.norm = norm
        
        # 损失权重
        self.adv_loss_weight = adv_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.fm_loss_weight = fm_loss_weight
        
        # 优化器参数
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_lr_scheduler = gen_lr_scheduler
        self.disc_lr_scheduler = disc_lr_scheduler
        
        # 训练参数
        self.disc_start_epoch = disc_start_epoch
        self.gen_train_freq = gen_train_freq
        self.disc_train_freq = disc_train_freq
        
        # 其他参数
        self.metrics = metrics
        self.val_metric = val_metric
        self.write_examples = write_examples
        self.sample_rate = sample_rate
        self.exp_name = exp_name
        self.reset = reset
        
        # 训练状态
        self.automatic_optimization = False  # 手动优化
        self.val_cpu_metric_input = []
        
        # 模型名称
        gen_name = type(generator).__name__
        disc_name = type(discriminator).__name__
        self.name = f"{gen_name}_{disc_name}_{exp_name}" if exp_name != 'exp' else f"{gen_name}_{disc_name}"

    def on_train_start(self):
        """训练开始时调用"""
        GS.on_train_start(
            self=self, 
            exp_name=self.exp_name, 
            model_name=self.name, 
            num_chns=1,  # 语音通常是单声道
            nfft=self.stft.n_fft, 
            model_class_path=self.import_path
        )

    def forward(self, x: Tensor, return_features: bool = False) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        生成器前向传播
        
        Args:
            x: 输入音频 [B, T] 或 [B, C, T]
            return_features: 是否返回中间特征
            
        Returns:
            生成的音频或(生成音频, 中间特征)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
            
        # STFT变换
        X, stft_paras = self.stft.stft(x)  # [B, C, F, T]
        B, C, F, T = X.shape
        
        # 标准化
        X, norm_paras = self.norm.norm(X)
        
        # 转换为实数表示
        X = X.permute(0, 2, 3, 1)  # [B, F, T, C]
        X = torch.view_as_real(X).reshape(B, F, T, -1)  # [B, F, T, 2C]
        
        # 生成器处理
        if return_features:
            gen_output, features = self.generator(X, return_features=True)
        else:
            gen_output = self.generator(X)
            features = None
            
        # 转换回复数
        if not torch.is_complex(gen_output):
            gen_output = torch.view_as_complex(gen_output.float().reshape(B, F, T, -1, 2))
            
        gen_output = gen_output.permute(0, 3, 1, 2)  # [B, Spk, F, T]
        
        # 逆标准化
        gen_output = self.norm.inorm(gen_output, norm_paras)
        
        # ISTFT变换
        gen_audio = self.stft.istft(gen_output, stft_paras)  # [B, Spk, T]
        
        if return_features:
            return gen_audio, features
        return gen_audio

    def discriminator_forward(self, x: Tensor, return_features: bool = False) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        判别器前向传播
        
        Args:
            x: 输入音频 [B, T] 或 [B, C, T]
            return_features: 是否返回中间特征
            
        Returns:
            判别结果或(判别结果, 中间特征)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
            
        if return_features:
            disc_output, features = self.discriminator(x, return_features=True)
            return disc_output, features
        else:
            disc_output = self.discriminator(x)
            return disc_output

    def adversarial_loss(self, disc_output: Tensor, is_real: bool) -> Tensor:
        """对抗损失"""
        if is_real:
            target = torch.ones_like(disc_output)
        else:
            target = torch.zeros_like(disc_output)
        return F.binary_cross_entropy_with_logits(disc_output, target)

    def feature_matching_loss(self, real_features: List[Tensor], fake_features: List[Tensor]) -> Tensor:
        """特征匹配损失"""
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
        return loss / len(real_features)

    def reconstruction_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """重构损失"""
        return self.recon_loss(pred, target)

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        x, y = batch  # x: 输入音频, y: 目标音频
        
        gen_opt, disc_opt = self.optimizers()
        
        # 当前是否训练判别器
        train_disc = (self.current_epoch >= self.disc_start_epoch and 
                     batch_idx % self.disc_train_freq == 0)
        train_gen = batch_idx % self.gen_train_freq == 0
        
        # 生成假音频
        fake_audio = self.forward(x)
        
        # 训练判别器
        if train_disc:
            disc_opt.zero_grad()
            
            # 真实音频的判别
            real_disc_output, real_features = self.discriminator_forward(y, return_features=True)
            real_loss = self.adversarial_loss(real_disc_output, is_real=True)
            
            # 假音频的判别
            fake_disc_output, fake_features = self.discriminator_forward(fake_audio.detach(), return_features=True)
            fake_loss = self.adversarial_loss(fake_disc_output, is_real=False)
            
            # 判别器总损失
            disc_loss = (real_loss + fake_loss) / 2
            
            self.manual_backward(disc_loss)
            disc_opt.step()
            
            # 记录判别器损失
            self.log('train/disc_loss', disc_loss, prog_bar=True, batch_size=x.shape[0])
            self.log('train/disc_real_loss', real_loss, batch_size=x.shape[0])
            self.log('train/disc_fake_loss', fake_loss, batch_size=x.shape[0])
        
        # 训练生成器
        if train_gen:
            gen_opt.zero_grad()
            
            # 重新生成（需要梯度）
            fake_audio = self.forward(x)
            
            # 对抗损失
            fake_disc_output, fake_gen_features = self.discriminator_forward(fake_audio, return_features=True)
            adv_loss = self.adversarial_loss(fake_disc_output, is_real=True)
            
            # 重构损失
            recon_loss = self.reconstruction_loss(fake_audio, y)
            
            # 特征匹配损失
            if train_disc:  # 只有在训练判别器时才有真实特征
                fm_loss = self.feature_matching_loss(real_features, fake_gen_features)
            else:
                fm_loss = torch.tensor(0.0, device=self.device)
            
            # 生成器总损失
            gen_loss = (self.adv_loss_weight * adv_loss + 
                       self.recon_loss_weight * recon_loss + 
                       self.fm_loss_weight * fm_loss)
            
            self.manual_backward(gen_loss)
            gen_opt.step()
            
            # 记录生成器损失
            self.log('train/gen_loss', gen_loss, prog_bar=True, batch_size=x.shape[0])
            self.log('train/gen_adv_loss', adv_loss, batch_size=x.shape[0])
            self.log('train/gen_recon_loss', recon_loss, batch_size=x.shape[0])
            self.log('train/gen_fm_loss', fm_loss, batch_size=x.shape[0])

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        x, y = batch
        
        # 生成音频
        fake_audio = self.forward(x)
        
        # 计算各种损失（仅用于监控）
        with torch.no_grad():
            # 重构损失
            recon_loss = self.reconstruction_loss(fake_audio, y)
            
            # 判别器输出
            fake_disc_output = self.discriminator_forward(fake_audio)
            real_disc_output = self.discriminator_forward(y)
            
            # 对抗损失
            gen_adv_loss = self.adversarial_loss(fake_disc_output, is_real=True)
            disc_real_loss = self.adversarial_loss(real_disc_output, is_real=True)
            disc_fake_loss = self.adversarial_loss(fake_disc_output, is_real=False)
            
            # 总损失
            gen_loss = self.adv_loss_weight * gen_adv_loss + self.recon_loss_weight * recon_loss
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            
            # 计算音频指标 (SDR, SI-SDR)
            # 确保音频维度正确 [B, Spk, T] 或 [B, T]
            if fake_audio.dim() == 3:
                fake_flat = fake_audio.squeeze(1)  # [B, T]
                y_flat = y.squeeze(1) if y.dim() == 3 else y  # [B, T]
            else:
                fake_flat = fake_audio  # [B, T]
                y_flat = y  # [B, T]
            
            # 计算SDR和SI-SDR
            sdr_val = sdr(fake_flat, y_flat).mean()
            si_sdr_val = si_sdr(preds=fake_flat, target=y_flat).mean()
        
        # 记录验证损失
        self.log('val/gen_loss', gen_loss, sync_dist=True, batch_size=x.shape[0])
        self.log('val/disc_loss', disc_loss, sync_dist=True, batch_size=x.shape[0])
        self.log('val/recon_loss', recon_loss, sync_dist=True, batch_size=x.shape[0])
        
        # 记录音频指标
        self.log('val/sdr', sdr_val, sync_dist=True, batch_size=x.shape[0])
        self.log('val/si_sdr', si_sdr_val, sync_dist=True, batch_size=x.shape[0])
        
        # 选择验证指标
        val_metric_value = {
            'gen_loss': gen_loss, 
            'recon_loss': recon_loss, 
            'disc_loss': disc_loss,
            'sdr': sdr_val,
            'si_sdr': si_sdr_val
        }[self.val_metric]
        self.log('val/metric', val_metric_value, sync_dist=True, batch_size=x.shape[0])
        
        # 保存样本用于日志记录
        if batch_idx == 0:
            self._input_audio = x[0]
            self._target_audio = y[0] 
            self._generated_audio = fake_audio[0]

    def on_validation_epoch_end(self) -> None:
        """验证epoch结束时调用"""
        # 这里可以添加音频样本的记录逻辑
        pass

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        x, y, paras = batch
        
        # 生成音频
        fake_audio = self.forward(x)
        
        # 计算评估指标
        sample_rate = paras[0].get('sample_rate', self.sample_rate)
        
        # 使用现有的指标计算函数
        if hasattr(self, 'metrics') and self.metrics:
            metrics, _, _ = cal_metrics_functional(
                self.metrics, 
                fake_audio[0], 
                y[0], 
                x[0].expand_as(y[0]), 
                sample_rate, 
                device_only='gpu'
            )
        else:
            metrics = {}
        
        # 保存结果
        wavname = os.path.basename(f"{paras[0].get('index', batch_idx)}.wav")
        result_dict = {
            'id': batch_idx,
            'wavname': wavname,
            **metrics
        }
        
        # 写入音频样本
        if self.write_examples < 0 or batch_idx < self.write_examples:
            exp_save_path = self.trainer.logger.log_dir
            os.makedirs(exp_save_path, exist_ok=True)
            
            # 保存输入、目标和生成的音频
            import torchaudio
            torchaudio.save(
                os.path.join(exp_save_path, f"input_{wavname}"),
                x[0].cpu(), sample_rate
            )
            torchaudio.save(
                os.path.join(exp_save_path, f"target_{wavname}"),
                y[0].cpu(), sample_rate
            )
            torchaudio.save(
                os.path.join(exp_save_path, f"generated_{wavname}"),
                fake_audio[0].cpu(), sample_rate
            )
        
        return result_dict

    def predict_step(self, batch: Union[Tensor, Tuple], batch_idx: Optional[int] = None) -> Tensor:
        """预测步骤"""
        if isinstance(batch, Tensor):
            x = batch
        else:
            x = batch[0]
        
        # 生成音频
        generated_audio = self.forward(x)
        
        # 标准化输出
        max_vals = torch.max(torch.abs(generated_audio), dim=-1).values
        norm = torch.where(max_vals > 1, max_vals, 1)
        generated_audio = generated_audio / norm.unsqueeze(-1)
        
        return generated_audio

    def configure_optimizers(self):
        """配置优化器"""
        # 生成器优化器
        gen_opt_class = getattr(torch.optim, self.gen_optimizer[0])
        gen_optimizer = gen_opt_class(self.generator.parameters(), **self.gen_optimizer[1])
        
        # 判别器优化器  
        disc_opt_class = getattr(torch.optim, self.disc_optimizer[0])
        disc_optimizer = disc_opt_class(self.discriminator.parameters(), **self.disc_optimizer[1])
        
        optimizers = [gen_optimizer, disc_optimizer]
        
        # 学习率调度器
        schedulers = []
        if self.gen_lr_scheduler is not None:
            gen_sched_class = getattr(torch.optim.lr_scheduler, self.gen_lr_scheduler[0])
            gen_scheduler = gen_sched_class(gen_optimizer, **self.gen_lr_scheduler[1])
            schedulers.append(gen_scheduler)
            
        if self.disc_lr_scheduler is not None:
            disc_sched_class = getattr(torch.optim.lr_scheduler, self.disc_lr_scheduler[0])
            disc_scheduler = disc_sched_class(disc_optimizer, **self.disc_lr_scheduler[1])
            schedulers.append(disc_scheduler)
        
        if schedulers:
            return optimizers, schedulers
        return optimizers

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """加载检查点时调用"""
        if hasattr(GS, 'on_load_checkpoint'):
            GS.on_load_checkpoint(
                self=self, 
                checkpoint=checkpoint, 
                ensemble_opts=None, 
                compile=False, 
                reset=self.reset
            )


class GANTrainCLI(BaseCLI):
    """GAN训练CLI"""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # 基本训练配置
        parser.set_defaults({"trainer.strategy": "ddp_find_unused_parameters_false"})
        parser.set_defaults({"trainer.accelerator": "gpu"})
        
        # EarlyStopping
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        early_stopping_defaults = {
            "early_stopping.enable": False,
            "early_stopping.monitor": "val/metric",
            "early_stopping.patience": 20,
            "early_stopping.mode": "min",  # GAN通常监控损失，越小越好
            "early_stopping.min_delta": 0.01,
        }
        parser.set_defaults(early_stopping_defaults)

        # ModelCheckpoint
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_metric{val/metric:.4f}",
            "model_checkpoint.monitor": "val/metric",
            "model_checkpoint.mode": "min",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 3,  # GAN保存最好的几个
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True
        }
        parser.set_defaults(model_checkpoint_defaults)

        # 最新检查点
        parser.add_lightning_class_args(ModelCheckpoint, "last_checkpoint")
        last_checkpoint_defaults = {
            "last_checkpoint.filename": "last",
            "last_checkpoint.monitor": "epoch",
            "last_checkpoint.mode": "max",
            "last_checkpoint.every_n_epochs": 1,
            "last_checkpoint.save_top_k": 1,
            "last_checkpoint.save_last": False
        }
        parser.set_defaults(last_checkpoint_defaults)

        # 添加模型无关的参数
        if hasattr(self, 'add_model_invariant_arguments_to_parser'):
            self.add_model_invariant_arguments_to_parser(parser)


if __name__ == '__main__':
    # python Trainer.py --help
    cli = GANTrainCLI(
        GANTrainModule,
        pl.LightningDataModule,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        subclass_mode_data=True,
    )