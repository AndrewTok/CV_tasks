import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
import pandas as pd
import os
import torchvision
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



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
            # transformed = self.transform()
        return transformed

    def __getitem__(self, index):
        filename, labels = self.items[index]
        img_path = os.path.join(self.imgs_path, filename)
        ## read image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image).astype(np.float32)
        
        ## augmentation
        if self.transform:
            image = self.transform(image=image)["image"]
        
        ## to Tensor
        x = torch.from_numpy(image).permute(2, 0, 1)
        # print(image.shape)
        x_scale = 100/image.shape[1]
        y_scale = 100/image.shape[0]
        x = torchvision.transforms.functional.resize(x, (100,100), antialias=True)
        if labels is not None:
            labels = self.transform_labels(labels, x_scale, y_scale)

        return x, labels


class DetectorBlock(torch.nn.Module):


    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
    def forward(self, x):
        return self.block(x)

class Flattener(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        batch_size, *_ = x.shape
        res = x.view(batch_size, -1) # torch.flatten(x) #
        return res

class FacesPointsDetector(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            DetectorBlock(3, 64, 7),
            DetectorBlock(64, 128, 5),
            DetectorBlock(128, 256, 3),
            DetectorBlock(256, 512, 3),
            Flattener(),
            nn.LazyLinear(out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.LazyLinear(out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.LazyLinear(out_features=28)
        )


    def forward(self, x):
        return self.layers(x)




class FacesPointsTrainingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = FacesPointsDetector()
        self.loss_f = F.mse_loss 
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
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4) #, weight_decay=5e-4

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = 8,
            gamma = 1,
            verbose = True
        )
        # torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=0.2,
        #     patience=5,
        #     verbose=True,
        #     # eps = 1e-1,
        # )
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

    # OPTIONAL
    def validation_step(self, batch, batch_idx):
        """the full validation loop"""
        x, y_gt = batch

        y_pr = self.model(x)
        loss = self.loss_f(y_pr, y_gt.float())

        metrics = {"val_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)


    def forward(self, x):
        return self.model(x)


def train_detector(train_gt, train_img_dir, fast_train=True, val_loader = None):
    '''
    Возвращает словарь размером N, ключи которого — имена файлов, а значения —
    массив чисел длины 28 с координатами точек лица [x1, y1, . . . , x14, y14]. Здесь N — количество
    изображений.
    
    '''
    ## Save the training module periodically by monitoring a quantity.
    # MyTrainingModuleCheckpoint = ModelCheckpoint(
    #     dirpath="runs/faces_points_detector",
    #     filename="{epoch}-{val_acc:.3f}",
    #     monitor="val_loss",
    #     mode="min",
    #     save_top_k=1,
    # )

    train_dataset = FacesPointsDataset(train_img_dir, train_gt, 'train')
    train_loader = DataLoader(train_dataset, 32, drop_last=True)
    training_module = FacesPointsTrainingModule()
    if not fast_train:
        trainer = pl.Trainer(accelerator='cuda', devices=1, max_epochs=32)
    else:
        trainer = pl.Trainer(accelerator='cpu', devices=0, max_epochs=1)
    
    trainer.fit(training_module, train_loader, val_dataloaders=val_loader)
    
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
    train_detector()