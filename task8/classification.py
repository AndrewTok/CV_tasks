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
import pytorch_lightning as pl
import albumentations as A







_target_image_size = 224

class BirdsClassificationDataset(Dataset):


    def __init__(self,
        imgs_path: str,
        labels: str,
        mode: str,
        fraction = 0.8,
        transform = None,
    ):
        self.imgs_path = imgs_path
        self.mode = mode
        self.fraction = fraction
        self.transform = transform

        self.labels = labels

        self.items = []

        imgs = os.listdir(imgs_path)
        imgs_num = len(imgs)
        permutation = np.random.RandomState(seed=42).permutation(imgs_num)

        if mode == 'train':
            start = 0
            end = int(fraction*imgs_num)
        elif mode == 'val':
            start = int(fraction*imgs_num)
            end = imgs_num
        elif mode == 'fast_train':
            start = 0
            end = 5
        elif mode == 'test':
            start = 0
            end = imgs_num
        else:
            raise Exception(f'wrong dataset mode: {mode}')

        for j in range(start, end):
            i = permutation[j]
            if self.labels is not None:
                labels = self.labels[imgs[i]]
                labels = labels  #int(labels[0]) #torch.from_numpy(labels)
                self.items.append((imgs[i], labels))
            else:
                self.items.append((imgs[i], None))



    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        filename, labels = self.items[index]
        img_path = os.path.join(self.imgs_path, filename)
        ## read image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image).astype(np.float32)
        
        ## augmentation
        if self.transform:
            image = self.transform(image=image)['image']

        


        x_scale = _target_image_size/image.shape[1]
        y_scale = _target_image_size/image.shape[0]

        # self.last_x_scale = x_scale
        # self.last_y_scale = y_scale

        ## to Tensor
        x = torch.from_numpy(image).permute(2, 0, 1)

        x = torchvision.transforms.functional.resize(x, (_target_image_size,_target_image_size), antialias=True)

        return x, labels



class BirdsClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        imgs_dir,
        labels,
        batch_size = 16,
        fraction = 0.8,
        train_transform=None,
        val_transform = None
    ):
        super().__init__()

        self.batch_size = batch_size
        self.imgs_dir = imgs_dir
        self.labels = labels


        self.train_set = BirdsClassificationDataset(imgs_dir, labels, 'train', fraction, train_transform)
        self.valid_set = BirdsClassificationDataset(imgs_dir, labels, 'val', fraction, val_transform)


    # def setup(self, stage):
    #     print(f"Train: {len(self.train_set)} images\nValidation: {len(self.valid_set)} images\n")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            drop_last=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            drop_last=True
        )

    # def test_dataloader(self):
    #     test_set = BirdsClassificationDataset(imgs_path=self.imgs_dir, labels=None, mode='test', fraction=1, transform=None)
    #     return torch.utils.data.DataLoader(
    #         test_set,
    #         batch_size=self.batch_size,
    #     )

    def fast_train_dataloader(self):
        fast_train_set = BirdsClassificationDataset(self.imgs_dir, self.labels, 'fast_train')
        return torch.utils.data.DataLoader(
            fast_train_set,
            batch_size=self.batch_size,
            drop_last=True
        )


class MobileNetClassifier(nn.Module):


    def __init__(self, num_classes: int,  init_weights, unfreeze: int = 18):
        # super().___init___()
        super().__init__()

        if init_weights:
            self.net = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.DEFAULT)
        else:
            self.net = torchvision.models.convnext_small()

        linear_size = self.net.classifier[-1].in_features
        # self.net.classifier[-1] = nn.LazyLinear(512) 
        self.additional_layer = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.GELU(),
            nn.LazyLinear(512),

            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.LazyLinear(num_classes)
        )
        # = nn.Sequential(
        #     nn.Flatten(1), nn.Linear(linear_size, num_classes), nn.Softmax(1)
        # )


        for child in list(self.net.children()):
            for param in child.parameters():
                param.requires_grad = True
            
        for child in list(self.net.children())[:-unfreeze]:
            for param in child.parameters():
                param.requires_grad = False
            
    
    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(self.additional_layer(x), dim=1)



class BaseBirdsClassifierTrainingModule(pl.LightningModule):
    def __init__(self, fast_train = False):
        super().__init__()
        self.fast_train = fast_train

        self.model = MobileNetClassifier(num_classes=50, init_weights= not fast_train)

        self.loss_f =  F.nll_loss
        self.metric = lambda logits, y: torch.sum(logits.argmax(axis=1) == y)/ y.shape[0] # accuracy
        self.train_acc = []
        self.val_acc = []

        self.lr =  5e-4 #1e-3
        self.weight_decay = 1e-8
        self.best_val_acc = 0

    def training_step(self, batch, batch_idx):
        x, y_gt = batch
        y_pr = self.model(x)



        loss = self.loss_f(y_pr, y_gt) #y_gt.float()

        acc = self.metric(y_pr, y_gt)

        metrics = {'loss':loss.detach(), 'accuracy': acc.detach()}
        if not self.fast_train:
            self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.train_acc.append(acc.detach())  # optional


        return loss

    def configure_optimizers(self):
        """Define optimizers and LR schedulers."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay) #, 
        
        if not self.fast_train:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=5,
                verbose=True,
                threshold= 1e-2
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
        else:
            return optimizer
        # return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-3)

    ## OPTIONAL:
    def on_train_epoch_end(self):
        ## display average loss across epoch
        avg_loss = torch.stack(self.train_acc).mean()
        # print(
        #     f"Epoch {self.trainer.current_epoch}, "
        #     f"Train_acc: {round(float(avg_loss), 3)}",
        # )
        # don't forget to clear the saved losses
        self.train_acc.clear()

    def validation_step(self, batch, batch_idx):
        x, y_gt = batch
        y_pr = self.model(x)
        loss = self.loss_f(y_pr, y_gt) #.float()

        acc = self.metric(y_pr, y_gt)

        metrics = {'val_accuracy': acc.detach()}
        if not self.fast_train:
            self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        
        self.val_acc.append(acc.detach()) 
        

    def on_validation_end(self):
        avg_acc = torch.stack(self.val_acc).mean()
        # print(
        #     f"Epoch {self.trainer.current_epoch}, "
        #     f"Val_acc: {round(float(avg_loss), 3)}",
        # )
        if not self.fast_train:
            if avg_acc > self.best_val_acc:
                self.best_val_acc = avg_acc
                torch.save(self.model.state_dict(), 'best_current.pt')
        self.val_acc.clear()

    def forward(self, x):
        return self.model(x)



class BirdsClassifierTrainingModule(BaseBirdsClassifierTrainingModule):
    def __init__(self, fast_train = False):
        super(BirdsClassifierTrainingModule, self).__init__(fast_train)

    def configure_optimizers(self):
        params = list(self.model.named_parameters())

        grouped_parameters = [
            {"params": [p for _, p in params[:-3]], "lr": self.lr/20},
            {"params": [p for _, p in params[-3:]], "lr": self.lr}
        ]

        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.lr, weight_decay=self.weight_decay)

        if not self.fast_train:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=20,
                verbose=True,
                threshold= 1e-2
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
        else:
            return optimizer


MEAN = [124.4789401,  130.49218858, 118.92390399]
STD = [42.41922657, 42.41256829, 45.22351659]
default_transform = A.Compose(
    [
        # A.ToFloat(max_value = 255),
        # A.augmentations.geometric.resize.Resize(_target_image_size, _target_image_size),
        # A.Normalize(),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=70, p=0.3),
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.HorizontalFlip(p = 0.3),
        # A.VerticalFlip(p = 0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(), #mean=MEAN, std=STD max_pixel_value=1
    ]
)


test_transform = A.Compose(
    [
        # A.ToFloat(max_value = 255),
        # A.augmentations.geometric.resize.Resize(_target_image_size, _target_image_size),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        A.Normalize(),#mean=MEAN, std=STD max_pixel_value=1
    ]
)


def train_classifier(gt, img_dir, fast_train = True):

    if fast_train:
        mode = 'fast_train'
        batch_size = 2
    else:
        mode = 'train'
        batch_size = 16
		
    data_module = BirdsClassificationDataModule(img_dir, gt, batch_size=batch_size, fraction=0.8, train_transform=default_transform, val_transform = test_transform)

    training_module = BirdsClassifierTrainingModule(fast_train)

    if not fast_train:
        trainer = pl.Trainer(accelerator='cuda', devices=1, max_epochs=60)
        val_loader = data_module.val_dataloader()
        train_loader = data_module.train_dataloader()
    else:
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1, logger=False, enable_checkpointing=False)
        val_loader = None
        train_loader = data_module.fast_train_dataloader()
    
    trainer.fit(training_module, train_loader, val_dataloaders=val_loader)

    return training_module.model



def classify(model_ckpt, test_dir):

    # module = BirdsClassifierTrainingModule.load_from_checkpoint(model_ckpt, map_location='cpu', fast_train=False)
    
    model = MobileNetClassifier(num_classes=50, init_weights=False)
    model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))

    dataset = BirdsClassificationDataset(test_dir, None, mode='test', transform=test_transform)

    result = {}

    model.eval()

    for i in range(len(dataset)):
        img, _ = dataset[i]
        pred = model(img[None,:])
        result[dataset.items[i][0]] = np.argmax(pred.detach().numpy(), axis = 1)
        if i % 10 == 0:
            print(float(i) / len(dataset), np.argmax(pred.detach().numpy(), axis = 1))
    
    return result