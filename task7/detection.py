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



class FacesPointsDataset(Dataset):


    def __init__(self,
        imgs_path: str,
        labels_path: str,
        mode: str,
        fraction = 0.8,
        transform = None,
    ):
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.mode = mode
        self.fraction = fraction
        self.transform = transform

        self.labels_df = pd.read_csv(labels_path)

        self.items = []

        imgs = os.listdir(imgs_path)
        imgs_num = len(imgs)
        
        if mode == 'train':
            start = 0
            end = int(fraction*imgs_num)
        else:
            start = int(fraction*imgs_num)
            end = imgs_num

        for i in range(start, end):
            img = imgs[i]
            labels = self.labels_df.loc[self.labels_df['filename'] == img][self.labels_df.columns[1:]].values
            labels = torch.from_numpy(labels)
            self.items.append((os.path.join(self.imgs_path, img), labels[0]))

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

        return transformed

    def __getitem__(self, index):
        img_path, labels = self.items[index]

        ## read image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image).astype(np.float32)
        
        ## augmentation
        if self.transform:
            image = self.transform(image=image)["image"]
        
        ## to Tensor
        x = torch.from_numpy(image).permute(2, 0, 1)
        x_scale = 100/image.shape[1]
        y_scale = 100/image.shape[0]
        x = torchvision.transforms.functional.resize(x, (100,100), antialias=True)
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
            DetectorBlock(3, 64, 3),
            DetectorBlock(64, 128, 3),
            DetectorBlock(128, 256, 3),
            Flattener(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=28)
        )


    def forward(self, x):
        return self.layers(x)




class FacesPointsTrainingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = FacesPointsDetector()
        self.train_loss = []

    def training_step(self, batch, batch_idx):
        x, y_gt = batch
        y_pr = self.model(x)
        loss = F.mse_loss(y_pr, y_gt.float())

        metrics = {'loss':loss.detach()}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.train_loss.append(loss.detach())  # optional


        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)

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




def train_detector(fast_train=True):
    '''
    Возвращает словарь размером N, ключи которого — имена файлов, а значения —
    массив чисел длины 28 с координатами точек лица [x1, y1, . . . , x14, y14]. Здесь N — количество
    изображений.
    
    '''
    train_dataset = FacesPointsDataset('tests/00_test_img_input/train/images/', 'tests/00_test_img_input/train/gt.csv', 'train')
    train_loader = DataLoader(train_dataset, 16)
    training_module = FacesPointsTrainingModule()
    trainer = pl.Trainer(accelerator='cuda', devices=1, max_epochs=16)

    trainer.fit(training_module, train_loader)
    
    pass



def detect():



    pass



if __name__ == '__main__':
    train_detector()