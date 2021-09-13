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
import glob
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from nnAudio.Spectrogram import CQT1992v2, CQT2010v2
from scipy import signal
####################
# Utils
####################
def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
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
             'height': 512,#640,
             'width': 512,
             'model_name': 'efficientnet_b0',
             'lr': 0.001,
             'drop_rate': 0.0,
             'drop_path_rate': 0.0,
             'data_dir': '../input/seti-breakthrough-listen',
             'model_path': None,
             'output_dir': './',
             'seed': 2021,
             'snap': 1}
conf_base = OmegaConf.create(conf_dict)

####################
# Dataset
####################

class G2NetDataset(Dataset):
    def __init__(self, df, transform=None, conf=None, train=True):
        self.df = df.reset_index(drop=True)
        self.dir_names = df['dir'].values
        self.labels = df['target'].values
        self.wave_transform = [
            CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=8, bins_per_octave=8, window='flattop'),
            CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=8, bins_per_octave=8, window='blackmanharris'),
            CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=8, bins_per_octave=8, window='nuttall')]
        #self.wave_transform = CQT1992v2(sr=2048, fmin=10, fmax=1024, hop_length=8, bins_per_octave=8, window='flattop')
        #self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=1, bins_per_octave=14, window='flattop')
        #self.wave_transform = CQT2010v2(sr=2048, fmin=10, fmax=1024, hop_length=32, n_bins=32, bins_per_octave=8, window='flattop')
        self.stat = [
            [0.013205823003608798,0.037445450696502146],
            [0.009606230606511236,0.02489221471650526], # 10000 sample
            [0.009523397709568962,0.024628402379527688],] # 10000 sample
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
        #waves = waves / np.max(np.abs(waves), axis=1, keepdims=True)
        #waves = waves / np.max(waves)
        waves = waves / 4.6152116213830774e-20
        waves = torch.from_numpy(waves).float()
        image = transform(waves)
        return image

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id']
        file_path = os.path.join(self.dir_names[idx],"{}/{}/{}/{}.npy".format(img_id[0], img_id[1], img_id[2], img_id))
        waves = np.load(file_path)
        label = torch.tensor([self.labels[idx]]).float()

        image1 = self.apply_qtransform(waves, self.wave_transform[0])
        image1 = image1.squeeze().numpy().transpose(1,2,0)
        image1 = cv2.vconcat([image1[:,:,0],image1[:,:,1],image1[:,:,2]])
        image1 = (image1-self.stat[0][0])/self.stat[0][1]
        image1 = cv2.resize(image1, (self.conf.width, self.conf.height), interpolation=cv2.INTER_CUBIC)

        image2 = self.apply_qtransform(waves, self.wave_transform[1])
        image2 = image2.squeeze().numpy().transpose(1,2,0)
        image2 = cv2.vconcat([image2[:,:,0],image2[:,:,1],image2[:,:,2]])
        image2 = (image2-self.stat[1][0])/self.stat[1][1]
        image2 = cv2.resize(image2, (self.conf.width, self.conf.height), interpolation=cv2.INTER_CUBIC)

        image3 = self.apply_qtransform(waves, self.wave_transform[2])
        image3 = image3.squeeze().numpy().transpose(1,2,0)
        image3 = cv2.vconcat([image3[:,:,0],image3[:,:,1],image3[:,:,2]])
        image3 = (image3-self.stat[2][0])/self.stat[2][1]
        image3 = cv2.resize(image3, (self.conf.width, self.conf.height), interpolation=cv2.INTER_CUBIC)

        #if self.transform is not None:
        #    image = self.transform(image=image)['image']
        image1 = torch.from_numpy(image1).unsqueeze(dim=0)
        image2 = torch.from_numpy(image2).unsqueeze(dim=0)
        image3 = torch.from_numpy(image3).unsqueeze(dim=0)

        return image1, image2, image3, label
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
    def setup(self, stage=None, fold=None):
        if stage == 'test':
            #test_df = pd.read_csv(os.path.join(self.conf.data_dir, "sample_submission.csv"))
            #test_df['dir'] = os.path.join(self.conf.data_dir, "test")
            #self.test_dataset = G2NetDataset(test_df, transform=None,conf=self.conf, train=False)
            df = pd.read_csv(os.path.join(self.conf.data_dir, "training_labels.csv"))
            df['dir'] = os.path.join(self.conf.data_dir, "train")
            
            # cv split
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.conf.seed)
            for n, (train_index, val_index) in enumerate(skf.split(df, df['target'])):
                df.loc[val_index, 'fold'] = int(n)
            df['fold'] = df['fold'].astype(int)
            
            train_df = df[df['fold'] != fold]
            self.valid_df = df[df['fold'] == fold]

            self.valid_dataset = G2NetDataset(self.valid_df, transform=None,conf=self.conf, train=False)
        
# ====================================================
# Inference function
# ====================================================
def inference(models, test_loader):
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    with torch.no_grad():
      for i, (images) in tk0:
          images1 = images[0].cuda()
          images2 = images[1].cuda()
          images3 = images[2].cuda()
          avg_preds = []
          for model in models:
              y_preds = model(images1)/3.0
              y_preds += model(images2)/3.0
              y_preds += model(images3)/3.0

              avg_preds.append(y_preds.sigmoid().to('cpu').numpy())

          avg_preds = np.mean(avg_preds, axis=0)
          probs.append(avg_preds)
      probs = np.concatenate(probs)
    return probs
  
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(2021)

    # get model path
    model_path = []
    for i in range(5):
        target_model = glob.glob(os.path.join(conf.model_dir, f'fold{i}/ckpt/*epoch*.ckpt'))
        scores = [float(os.path.splitext(os.path.basename(i))[0].split('=')[-1]) for i in target_model]
        model_path.append(target_model[scores.index(max(scores))])
        
    models = []
    for ckpt in model_path:
      m = timm.create_model(model_name=conf.model_name, num_classes=1, pretrained=False, in_chans=1)
      m = load_pytorch_model(ckpt, m, ignore_suffix='model')
      m.cuda()
      m.eval()
      models.append(m)

    
    # make oof
    oof_df = pd.DataFrame()
    for f, m in enumerate(models):
        data_module = SETIDataModule(conf)
        data_module.setup(stage='test', fold=f)
        valid_df = data_module.valid_df
        valid_dataset = data_module.valid_dataset
        valid_loader =  DataLoader(valid_dataset, batch_size=conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
        
        predictions = inference([m], valid_loader)
        valid_df['preds'] = predictions
        oof_df = pd.concat([oof_df, valid_df])



        #test = pd.read_csv(os.path.join(conf.data_dir, "sample_submission.csv"))
        #test['target'] = predictions
    oof_df[['id', 'target', 'preds']].to_csv(os.path.join(conf.output_dir, "oof.csv"), index=False)
        
    print(oof_df[['id', 'target']].head())
    print(model_path)
    
    

if __name__ == "__main__":
    main()