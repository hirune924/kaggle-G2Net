####################
# Import Libraries
####################
import os
import sys
from PIL import Image
import cv2
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
import albumentations as A
import timm
from omegaconf import OmegaConf

from sklearn.metrics import roc_auc_score
from nnAudio.Spectrogram import CQT1992v2, CQT2010v2
from scipy import signal
####################
# Utils
####################
def get_score(y_true, y_pred):
    try:
        score = roc_auc_score(y_true, y_pred)
    except:
        score = 0.0
    return score

def load_pytorch_model(ckpt_name, model, ignore_suffix='model'):
    state_dict = torch.load(ckpt_name, map_location='cpu')["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith(str(ignore_suffix)+"."):
            name = name.replace(str(ignore_suffix)+".", "", 1)  # remove `model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model
####################
# Config
####################

conf_dict = {'batch_size': 8,#32, 
             'epoch': 30,
             'height': 256,#640,
             'width': 256,
             'model_name': 'efficientnet_b0',
             'lr': 0.001,
             'fold': 0,
             'drop_rate': 0.2,
             'drop_path_rate': 0.2,
             'data_dir': '../input/g2net-gravitational-wave-detection/',
             'model_path': None,
             'output_dir': './',
             'pseudo': None,
             'seed': 2021,
             'trainer': {}}
conf_base = OmegaConf.create(conf_dict)

####################
# Dataset
####################

class G2NetDataset(Dataset):
    def __init__(self, df, transform=None, conf=None, train=True):
        self.df = df.reset_index(drop=True)
        self.dir_names = df['dir'].values
        self.labels = df['target'].values
        self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=8, bins_per_octave=8, window='flattop')
        #self.wave_transform = CQT1992v2(sr=2048, fmin=10, fmax=1024, hop_length=8, bins_per_octave=8, window='flattop')
        #self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=1, bins_per_octave=14, window='flattop')
        #self.wave_transform = CQT2010v2(sr=2048, fmin=10, fmax=1024, hop_length=32, n_bins=32, bins_per_octave=8, window='flattop')
        # hop lengthは変えてみたほうが良いかも
        self.transform = transform
        self.conf = conf
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def apply_qtransform(self, waves, transform):
        #print(waves.shape)
        #waves = np.hstack(waves)
        #print(np.max(np.abs(waves), axis=1))
        waves = waves / np.max(np.abs(waves), axis=1, keepdims=True)
        waves = torch.from_numpy(waves).float()
        image = transform(waves)
        return image

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id']
        file_path = os.path.join(self.dir_names[idx],"{}/{}/{}/{}.npy".format(img_id[0], img_id[1], img_id[2], img_id))
        waves1 = np.load(file_path)
        label1 = torch.tensor([self.labels[idx]]).float()


        if self.train:
            if torch.rand(1) < 0.50:
                indx = torch.randint(0,len(self.df),[1]).numpy()[0]
                img_id = self.df.loc[indx, 'id']
                file_path = os.path.join(self.dir_names[indx],"{}/{}/{}/{}.npy".format(img_id[0], img_id[1], img_id[2], img_id))
                waves2 = np.load(file_path)
                label2 = torch.tensor([self.labels[indx]]).float()

                #alpha = 32.0
                #lam = np.random.beta(alpha, alpha)
                #waves = waves1 * lam + waves2 * (1-lam)
                waves = waves1 + waves2
                label = label1 + label2 - (label1*label2)
            else:
                waves = waves1
                label = label1

            if torch.rand(1) < 0.50:
                waves =  np.roll(waves, np.random.randint(waves.shape[1]), axis=1)

        else:
            waves = waves1
            label = label1


        #bHP, aHP = signal.butter(1, (20,750), btype='bandpass', fs=2024)
        #waves = np.array([signal.filtfilt(bHP, aHP, w) for w in waves])

        image = self.apply_qtransform(waves, self.wave_transform)
        image = image.squeeze().numpy().transpose(1,2,0)

        image = cv2.vconcat([image[:,:,0],image[:,:,1],image[:,:,2]])

        #image = (image-np.mean(image, axis=(0,1),keepdims=True))/np.std(image, axis=(0,1),keepdims=True)
        #image = (image-np.mean(image, axis=1,keepdims=True))/np.std(image, axis=1,keepdims=True)
        image = (image-np.mean(image))/np.std(image)

        #img_pl = Image.fromarray(image).resize((self.conf.height, self.conf.width), resample=Image.BICUBIC)
        #image = np.array(img_pl)
        #image = cv2.resize(image, (self.conf.width, self.conf.height), interpolation=cv2.INTER_CUBIC)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        #image = torch.from_numpy(image.transpose(2,0,1))#.unsqueeze(dim=0)
        image = torch.from_numpy(image).unsqueeze(dim=0)

        return image, label
           
####################
# Data Module
####################

class SETIDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf  

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage=None):
        if stage == 'fit':
            df = pd.read_csv(os.path.join(self.conf.data_dir, "training_labels.csv"))
            df['dir'] = os.path.join(self.conf.data_dir, "train")
            
            # cv split
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.conf.seed)
            for n, (train_index, val_index) in enumerate(skf.split(df, df['target'])):
                df.loc[val_index, 'fold'] = int(n)
            df['fold'] = df['fold'].astype(int)
            
            train_df = df[df['fold'] != self.conf.fold]
            valid_df = df[df['fold'] == self.conf.fold]

            if self.conf.pseudo is not None:
                pseudo_df = pd.read_csv(self.conf.pseudo)
                #pseudo_df = pseudo_df[(pseudo_df['target']<0.05)|(pseudo_df['target']>0.95)]

                pseudo_df['dir'] = os.path.join(self.conf.data_dir, "test")

                train_df = pd.concat([train_df, pseudo_df])
            
            train_transform = A.Compose([
                        #A.Resize(height=self.conf.high, width=self.conf.width, interpolation=1), 
                        #A.Flip(p=0.5),
                        #A.VerticalFlip(p=0.5),
                        #A.HorizontalFlip(p=0.5),
                        #A.ShiftScaleRotate(p=0.5),
                        #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
                        #A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
                        #A.CLAHE(clip_limit=(1,4), p=0.5),
                        #A.OneOf([
                        #    A.OpticalDistortion(distort_limit=1.0),
                        #    A.GridDistortion(num_steps=5, distort_limit=1.),
                        #    A.ElasticTransform(alpha=3),
                        #], p=0.20),
                        #A.OneOf([
                        #    A.GaussNoise(var_limit=[10, 50]),
                        #    A.GaussianBlur(),
                        #    A.MotionBlur(),
                        #    A.MedianBlur(),
                        #], p=0.20),
                        #A.Resize(size, size),
                        #A.OneOf([
                        #    A.JpegCompression(quality_lower=95, quality_upper=100, p=0.50),
                        #    A.Downscale(scale_min=0.75, scale_max=0.95),
                        #], p=0.2),
                        #A.IAAPiecewiseAffine(p=0.2),
                        #A.IAASharpen(p=0.2),
                        A.Cutout(max_h_size=int(self.conf.height * 0.1), max_w_size=int(self.conf.width * 0.1), num_holes=5, p=0.5),
                        #A.Normalize()
                        ])

            #valid_transform = A.Compose([
            #            A.Resize(height=self.conf.high, width=self.conf.width, interpolation=1), 
            #            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
            #            ])

            #self.train_dataset = G2NetDataset(train_df, transform=train_transform,conf=self.conf)
            self.train_dataset = G2NetDataset(train_df, transform=None,conf=self.conf, train=True)
            self.valid_dataset = G2NetDataset(valid_df, transform=None, conf=self.conf, train=False)
            
        #elif stage == 'test':
        #    test_df = pd.read_csv(os.path.join(self.conf.data_dir, "sample_submission.csv"))
        #    test_df['dir'] = os.path.join(self.conf.data_dir, "test")
        #    test_transform = A.Compose([
        #                A.Resize(height=self.conf.height, width=self.conf.width, interpolation=1, always_apply=False, p=1.0),
        #                #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
        #                ])
        #    self.test_dataset = G2NetDataset(test_df, transform=test_transform, conf=self.conf)
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
        
####################
# Lightning Module
####################

class LitSystem(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        #self.conf = conf
        self.save_hyperparameters(conf)
        self.model = timm.create_model(model_name=self.hparams.model_name, num_classes=1, pretrained=True, in_chans=1,
                                       drop_rate=self.hparams.drop_rate, drop_path_rate=self.hparams.drop_path_rate)
        if self.hparams.model_path is not None:
            print(f'load model path: {self.hparams.model_path}')
            self.model = load_pytorch_model(self.hparams.model_path, self.model, ignore_suffix='model')
        self.criteria = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        '''
        if self.current_epoch < self.hparams.epoch*0.8:
            # mixup
            alpha = 1.0
            lam = np.random.beta(alpha, alpha)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size)
            x = lam * x + (1 - lam) * x[index, :]
            y = lam * y +  (1 - lam) * y[index]
            #y = y + y[index] - (y * y[index])
        '''
        
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        
        return {
            "val_loss": loss,
            "y": y,
            "y_hat": y_hat
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs]).cpu().detach().numpy()
        y_hat = torch.cat([x["y_hat"] for x in outputs]).cpu().detach().numpy()

        #preds = np.argmax(y_hat, axis=1)

        val_score = get_score(y, y_hat)

        self.log('avg_val_loss', avg_val_loss)
        self.log('val_score', val_score)

        
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(conf.seed)

    tb_logger = loggers.TensorBoardLogger(save_dir=os.path.join(conf.output_dir, 'tb_log/'))
    csv_logger = loggers.CSVLogger(save_dir=os.path.join(conf.output_dir, 'csv_log/'))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, 'ckpt/'), monitor='val_score', 
                                          save_last=True, save_top_k=5, mode='max', 
                                          save_weights_only=True, filename=f'fold{conf.fold}-'+'{epoch}-{val_score:.5f}')

    data_module = SETIDataModule(conf)

    lit_model = LitSystem(conf)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=conf.epoch,
        gpus=-1,
        amp_backend='native',
        amp_level='O2',
        precision=16,
        num_sanity_val_steps=10,
        val_check_interval=1.0,
        #sync_batchnorm=True,
        **conf.trainer
            )

    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    main()
