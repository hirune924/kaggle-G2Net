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
             'fold': 0,
             'drop_rate': 0.0,
             'drop_path_rate': 0.0,
             'data_dir': '../input/seti-breakthrough-listen',
             'model_path': None,
             'output_dir': './',
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
            CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=8, bins_per_octave=8, window='nuttall'),
            CWT(wavelet_width=8,fs=2048,lower_freq=20,upper_freq=1024,n_scales=384,stride=8)]
        #self.wave_transform = CQT1992v2(sr=2048, fmin=10, fmax=1024, hop_length=8, bins_per_octave=8, window='flattop')
        #self.wave_transform = CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=1, bins_per_octave=14, window='flattop')
        #self.wave_transform = CQT2010v2(sr=2048, fmin=10, fmax=1024, hop_length=32, n_bins=32, bins_per_octave=8, window='flattop')
        self.stat = [
            [0.013205823003608798,0.037445450696502146],
            [0.009606230606511236,0.02489221471650526], # 10000 sample
            [0.009523397709568962,0.024628402379527688],
            [0.0010164694150735158,0.0015815201992169022]] # 10000 sample
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

        image4 = self.apply_qtransform(waves, self.wave_transform[3])
        image4 = image4.squeeze().numpy().transpose(1,2,0)
        image4 = cv2.vconcat([image4[:,:,0],image4[:,:,1],image4[:,:,2]])
        image4 = (image4-self.stat[3][0])/self.stat[3][1]
        image4 = cv2.resize(image4, (self.conf.width, self.conf.height), interpolation=cv2.INTER_CUBIC)

        #if self.transform is not None:
        #    image = self.transform(image=image)['image']
        image1 = torch.from_numpy(image1).unsqueeze(dim=0)
        image2 = torch.from_numpy(image2).unsqueeze(dim=0)
        image3 = torch.from_numpy(image3).unsqueeze(dim=0)
        image4 = torch.from_numpy(image4).unsqueeze(dim=0)

        return image1, image2, image3, image4, label
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
        if stage == 'test':
            test_df = pd.read_csv(os.path.join(self.conf.data_dir, "sample_submission.csv"))
            test_df['dir'] = os.path.join(self.conf.data_dir, "test")
            self.test_dataset = G2NetDataset(test_df, transform=None,conf=self.conf, train=False)
        
# ====================================================
# Inference function
# ====================================================
def inference(models, test_loader):
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    raw_probs =  [[] for i in range(len(models))]
    probs = []
    probs_flattop = []
    probs_blackmanharris = []
    probs_nuttall = []
    probs_cwt = []
    with torch.no_grad():
      for i, (images) in tk0:
          images1 = images[0].cuda()
          images2 = images[1].cuda()
          images3 = images[2].cuda()
          images4 = images[3].cuda()
          avg_preds = []
          flattop = []
          blackmanharris = []
          nuttall = []
          cwt = []
          for mid, model in enumerate(models):
              y_preds_1 = model(images1)
              y_preds_2 = model(images2)
              y_preds_3 = model(images3)
              y_preds_4 = model(images4)
              y_preds   = (y_preds_1 + y_preds_2 + y_preds_3 + y_preds_4)/4

              avg_preds.append(y_preds.sigmoid().to('cpu').numpy())
              flattop.append(y_preds_1.sigmoid().to('cpu').numpy())
              blackmanharris.append(y_preds_2.sigmoid().to('cpu').numpy())
              nuttall.append(y_preds_3.sigmoid().to('cpu').numpy())
              cwt.append(y_preds_4.sigmoid().to('cpu').numpy())

              raw_probs[mid].append(y_preds.sigmoid().to('cpu').numpy())

          avg_preds = np.mean(avg_preds, axis=0)
          flattop = np.mean(flattop, axis=0)
          blackmanharris = np.mean(blackmanharris, axis=0)
          nuttall = np.mean(nuttall, axis=0)
          cwt = np.mean(cwt, axis=0)

          probs.append(avg_preds)
          probs_flattop.append(flattop)
          probs_blackmanharris.append(blackmanharris)
          probs_nuttall.append(nuttall)
          probs_cwt.append(cwt)
      
      for mid in range(len(models)):
          raw_probs[mid] = np.concatenate(raw_probs[mid])
      probs = np.concatenate(probs)
      probs_flattop = np.concatenate(probs_flattop)
      probs_blackmanharris = np.concatenate(probs_blackmanharris)
      probs_nuttall = np.concatenate(probs_nuttall)
      probs_cwt = np.concatenate(probs_cwt)
    return probs, probs_flattop, probs_blackmanharris, probs_nuttall, probs_cwt, raw_probs
  
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
        #if i == 4:
            #model_path.append('/kqi/parent/22021886/fold3_0/ckpt/fold3-epoch=18-val_score=0.91562.ckpt')
        #    continue
        #if i == 3:
        #    continue
        #for j in range(conf.snap):
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

    data_module = SETIDataModule(conf)
    data_module.setup(stage='test')
    test_dataset = data_module.test_dataset
    test_loader =  DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
    
    predictions, probs_flattop, probs_blackmanharris, probs_nuttall, probs_cwt, raw_probs = inference(models, test_loader)

    test = pd.read_csv(os.path.join(conf.data_dir, "sample_submission.csv"))
    for mid, rp in enumerate(raw_probs):
        test['target'] = rp
        test[['id', 'target']].to_csv(os.path.join(conf.output_dir, f'pseudo_fold{mid}.csv'), index=False)

    test['target'] = predictions
    test[['id', 'target']].to_csv(os.path.join(conf.output_dir, "submission.csv"), index=False)

    test['target'] = probs_flattop
    test[['id', 'target']].to_csv(os.path.join(conf.output_dir, "submission_flattop.csv"), index=False)
    test['target'] = probs_blackmanharris
    test[['id', 'target']].to_csv(os.path.join(conf.output_dir, "submission_blackmanharris.csv"), index=False)
    test['target'] = probs_nuttall
    test[['id', 'target']].to_csv(os.path.join(conf.output_dir, "submission_nuttall.csv"), index=False)
    test['target'] = probs_cwt
    test[['id', 'target']].to_csv(os.path.join(conf.output_dir, "submission_cwt.csv"), index=False)
    
    print(test[['id', 'target']].head())
    print(model_path)
    
    

if __name__ == "__main__":
    main()