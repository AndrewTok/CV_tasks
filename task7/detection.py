from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
import pandas as pd
import os
import torchvision
from PIL import Image
import albumentations as A
import pytorch_lightning as pl



class FacesPointsDataset(Dataset):


    def __init__(self,
        imgs_path: str,
        labels_path: str,
        mode: str,
        # fraction = 0.8,
        transform = None,
    ):
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.mode = mode
        # self.fraction = fraction
        self.transform = transform

        if labels_path is not None:
            self.labels_df = pd.read_csv(labels_path)
        else:
            self.labels_df = None

        self.items = []

        imgs = os.listdir(imgs_path)
        imgs_num = len(imgs)
        start = 0
        end = imgs_num        
        # print(self.labels_df.head())
        # if mode == 'train':
        #     start = 0
        #     end = int(fraction*imgs_num)
        # elif mode == 'val':
        #     start = int(fraction*imgs_num)
        #     end = imgs_num
        # else:
        #     start = 0
        #     end = imgs_num

        for i in range(start, end):
            # img = imgs[i]
            if self.labels_df is not None:
                # print(imgs[i])
                # print(self.labels_df.loc[i])
                labels = self.labels_df.loc[self.labels_df['filename'] == imgs[i]][self.labels_df.columns[1:]].values
                labels = torch.from_numpy(labels)
                
                if len(labels.shape) > 1:
                    self.items.append((imgs[i], labels[0]))
                else:
                    self.items.append((imgs[i], labels))

            else:
                self.items.append((imgs[i], None))

        # print(self.items[0][1].shape)



    def __len__(self):
        return len(self.items)

    def swap_axes(self, tensor, ax1, ax2):
        x = tensor[:, 1].clone()
        y = tensor[:, 0].clone()
        tensor[:, 0] = x
        tensor[:, 1] = y

    def get_label_xy_format(self, label):
        label = label.clone()
        label = label.reshape(14, 2)
        # swap_axes(label, 0, 1)
        # label[:, 0], label[:, 1] = label[:, 1], label[:, 0].clone()
        return label

    def get_label_original_format(self, label_xy):
        label = label_xy.clone()
        # swap_axes(label, 0, 1)
        label = label.reshape(28)
        # print(label.shape)
        return label

    def transform_labels(self, labels, x_scale, y_scale):
        '''
        resize image and label 
        '''

        transformed = labels.clone()
        for i in range(len(labels)):
            if i % 2 == 0:
                scale = x_scale
            else:
                scale = y_scale
            transformed[i] = int(transformed[i]*scale)
        return transformed



    def __getitem__(self, index):
        filename, labels = self.items[index]
        img_path = os.path.join(self.imgs_path, filename)
        ## read image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image).astype(np.float32)
        


        ## augmentation
        if self.transform:
            if labels is not None:
                keypoints = self.get_label_xy_format(labels)
                transformed = self.transform(image=image, keypoints=keypoints) #["image"]
                image = transformed['image']
                labels = torch.Tensor(transformed['keypoints'])            
                labels = self.get_label_original_format(labels)
            else:
                image = self.transform(image=image)['image']

        x_scale = 100/image.shape[1]
        y_scale = 100/image.shape[0]

        ## to Tensor
        x = torch.from_numpy(image).permute(2, 0, 1)
        # print(image.shape)

        x = torchvision.transforms.functional.resize(x, (100,100), antialias=True)
        if labels is not None:
            labels = self.transform_labels(labels, x_scale, y_scale)

        return x, labels


class DetectorBlock(torch.nn.Module):


    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, max_pool = True):
        super().__init__()
        self.max_pool = None
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
        )
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.block(x)
        if self.max_pool:
            x = self.max_pool(x)
        return x
    

class Flattener(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        batch_size, *_ = x.shape
        # print(batch_size)
        res = x.view(batch_size, -1) # torch.flatten(x) #
        return res

class FacesPointsDetector(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            DetectorBlock(3, 64, 7, True),
            # DetectorBlock(64, 64, 7, True),
            DetectorBlock(64, 128, 5, True),
            # DetectorBlock(128, 128, 5, True),
            DetectorBlock(128, 256, 3, True),
            # DetectorBlock(256, 256, 3, True),
            Flattener(),
            nn.LazyLinear(out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.LazyLinear(out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.LazyLinear(out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.LazyLinear(out_features=28)
        )


    def forward(self, x):
        return self.layers(x)


class FacePointsDetectorImproved(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            DetectorBlock(3, 64, 3, False),
            DetectorBlock(64, 64, 3, True),

            DetectorBlock(64, 128, 3, False),
            DetectorBlock(128, 128, 3, True),

            DetectorBlock(128, 256, 3, False),
            DetectorBlock(256, 256, 3, False),
            #DetectorBlock(256, 256, 3, False),
            DetectorBlock(256, 256, 3, True),

            DetectorBlock(256, 512, 3, False),
            DetectorBlock(512, 512, 3, False),
           # DetectorBlock(512, 512, 3, False),
            DetectorBlock(512, 512, 3, True),

            #DetectorBlock(512, 512, 3, False),
            #DetectorBlock(512, 512, 3, False),
            #DetectorBlock(512, 512, 3, False),
            #DetectorBlock(512, 512, 3, True),

            Flattener(),
            # nn.LazyLinear(out_features=4096),
            # nn.BatchNorm1d(4096),
            # nn.ReLU(),
            nn.LazyLinear(out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.LazyLinear(out_features=2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            #nn.LazyLinear(out_features=1024),
            #nn.BatchNorm1d(1024),
            #nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.LazyLinear(out_features=28)
        )


    def forward(self, x):
        return self.layers(x)


class FacesPointsTrainingModule(pl.LightningModule):
    def __init__(self, vgg = False):
        super().__init__()
        if vgg:
            self.model = FacePointsDetectorImproved()
        else:
            self.model = FacesPointsDetector()
        self.loss_f =  F.mse_loss
        self.train_loss = []

    def training_step(self, batch, batch_idx):
        x, y_gt = batch
        y_pr = self.model(x)
        loss = self.loss_f(y_pr, y_gt.float())

        metrics = {'loss':loss.detach()}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.train_loss.append(loss.detach())  # optional


        return loss

    def configure_optimizers(self):
        """Define optimizers and LR schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2) #3, weight_decay=5e-4

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size = 15,
        #     gamma = 1,
        #     verbose = True
        # )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=3,
            verbose=True,
            threshold=1e-2,
        )

        lr_dict = {
            # The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "loss",
        }

        return [optimizer], [lr_dict]
        # return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-3)

    ## OPTIONAL:
    def on_train_epoch_end(self):
        ## display average loss across epoch
        avg_loss = torch.stack(self.train_loss).mean()
        print(
            f"Epoch {self.trainer.current_epoch}, "
            f"Train_loss: {round(float(avg_loss), 3)}",
        )
        # don't forget to clear the saved losses
        self.train_loss.clear()

    def validation_step(self, batch, batch_idx):
        x, y_gt = batch
        y_pr = self.model(x)
        loss = self.loss_f(y_pr, y_gt.float())

        metrics = {'val_loss':loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        pass

    def forward(self, x):
        return self.model(x)

def train_detector(train_gt, train_img_dir, fast_train=True, val_loader = None):
    '''
    Возвращает словарь размером N, ключи которого — имена файлов, а значения —
    массив чисел длины 28 с координатами точек лица [x1, y1, . . . , x14, y14]. Здесь N — количество
    изображений.
    
    '''
    # MyTransform = A.Compose(
    # [
    #     # A.RandomResizedCrop(width=90, height=90, p=0.5),
    #     A.Rotate(limit=70),
    #     # A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=0.5), #brightness_limit=0.3, contrast_limit=0.3,
    #     A.ShiftScaleRotate(p=0.5)
    # ], 
    #     keypoint_params = A.KeypointParams(format = 'xy', remove_invisible=False) 
    # )
    train_dataset = FacesPointsDataset(train_img_dir, train_gt, 'train', transform=None)
    
    training_module = FacesPointsTrainingModule(False)
    if not fast_train:
        train_loader = DataLoader(train_dataset, 32, drop_last=True)
        trainer = pl.Trainer(accelerator='cuda', devices=1, max_epochs=64)
    else:
        train_loader = DataLoader(train_dataset, 1)
        trainer = pl.Trainer(accelerator='cpu', devices=0, max_epochs=1, fast_dev_run=True)
    
    trainer.fit(training_module, train_loader, val_loader)
    
    return training_module.model



def detect(model_filename, test_img_dir):

    
    module = FacesPointsTrainingModule.load_from_checkpoint(model_filename, map_location='cpu')

    dataset = FacesPointsDataset(test_img_dir, None, mode='test', transform=None)

    result = {}

    for i in range(len(dataset)):
        img, _ = dataset[i]
        pred = module[img[None,:]].detach()[0]
        result[dataset.items[i][0]] = pred
    
    return result



if __name__ == '__main__':
    #train_detector()
    pass