from Trainer import TrainModule
import yaml
import torch
from lightning.pytorch.cli import LightningArgumentParser
import pytorch_lightning as pl

def load_trainer_model_dm(config):
    r"""
    Load back trainer, model and datamodule after training.

    This function assumes that all we need is to perform inference.

    Parameters
    ----------
    config: str
        Path to the configuration file.
    """
    with open(config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # No need for inference.
    config_dict['trainer']['logger'] = False
    del config_dict['seed_everything'], config_dict['ckpt_path'], config_dict['data'], config_dict['early_stopping'], config_dict['model_checkpoint'], config_dict['optimizer'], config_dict['lr_scheduler']
    
    parser = LightningArgumentParser()
    parser.add_class_arguments(TrainModule, 'model', fail_untyped=False)
    # parser.add_class_arguments(pl.LightningDataModule, 'data', fail_untyped=False)
    parser.add_class_arguments(pl.Trainer, 'trainer', fail_untyped=False)
    config = parser.parse_object(config_dict)
    objects = parser.instantiate_classes(config)

    return objects.trainer, objects.model

if __name__ == '__main__':
    ckpt_path = '/home/imu_wutc/ddn/dataset/wtc/Vocex2/wtc/challege_code/logs/AVIGCRN_audio_only_exp/version_0/checkpoints/epoch54_metric12.2087.ckpt'
    trainer,model = load_trainer_model_dm('/home/imu_wutc/ddn/dataset/wtc/Vocex2/wtc/challege_code/logs/AVIGCRN_audio_only_exp/version_0/config.yaml')
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    torch.save(model.arch.state_dict(),'/home/imu_wutc/ddn/dataset/wtc/Vocex2/wtc/challege_code/audio_only.pt')
    print('lightning ckpt has transed to pytorch pt !')