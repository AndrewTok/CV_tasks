# -*- coding: utf-8 -*-
import torch
import torchvision
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
from torch.utils.data import DataLoader

import os
import csv
import json
import tqdm
import pickle
import typing

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier

from PIL import Image


CLASSES_CNT = 205

TARGET_IMAGE_SIZE = 224


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.
    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """
    def __init__(self, root_folders, path_to_classes_json) -> None:
        super(DatasetRTSD, self).__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        samples = []
        classes_to_samples = {id: [] for id in self.class_to_idx.values()}
        for root_folder in root_folders:
            class_names = os.listdir(root_folder)
            for class_name in class_names:
                id = self.class_to_idx[class_name]
                class_pth = root_folder + '/' + class_name
                imgs_names = os.listdir(class_pth)
                for filename in imgs_names:
                    img_path = class_pth + '/' + filename
                    
                    samples.append((img_path, id))
                    curr_pos = len(samples) - 1
                    classes_to_samples[id].append(curr_pos)
                    # if id in classes_to_samples.keys():
                    #
                    # else:
                    #     classes_to_samples[id] = [curr_pos]
            # for class_name in self.class_to_idx.keys():

                

        self.samples = samples ### YOUR CODE HERE - список пар (путь до картинки, индекс класса)
        self.classes_to_samples = classes_to_samples ### YOUR CODE HERE - cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        self.transform = A.Compose(
            [
                A.Resize(height=TARGET_IMAGE_SIZE, width=TARGET_IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(),
                A.pytorch.ToTensorV2()
            ]
        ) ### YOUR CODE HERE - аугментации + нормализация + ToTensorV2

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE
        img_path, id = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image).astype(np.float32)
        return self.transform(image=image)['image'], img_path, id


    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        :param path_to_classes_json: путь до classes.json
        """
        
        with open(path_to_classes_json, 'r') as f:
            r = f.read()
        d = eval(r)
        
        class_to_idx = {}
        
        classes = [] #['' for i in range(CLASSES_CNT)]

        for key in d.keys():
            id = int(d[key]['id']) 
            type = d[key]['type']
            class_to_idx[key] = id
            classes.append(key)

        # classes - class names array classes[id] = class_name
        return classes, class_to_idx

    def __len__(self):
        return len(self.samples)


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.
    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    def __init__(self, root, path_to_classes_json, annotations_file=None):
        super(TestData, self).__init__()

        classes, classes_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
        samples = []
        img_names = os.listdir(root)
        img_name_to_idx = {}
        for img_name in img_names:
            samples.append( img_name) #root + '/' +
            img_name_to_idx[img_name] = len(samples) - 1

        self.root = root
        self.samples = samples ### YOUR CODE HERE - список путей до картинок
        self.transform = A.Compose(
            [
                A.Resize(height=TARGET_IMAGE_SIZE, width=TARGET_IMAGE_SIZE),
                A.Normalize(),
                A.pytorch.ToTensorV2()
            ]
        ) ### YOUR CODE HERE - преобразования: ресайз + нормализация + ToTensorV2
        self.targets = None
        if annotations_file is not None:
            targets = {}
            annotation = pd.read_csv(annotations_file)
            for i in range(len(annotation)):
                img_name, class_id = annotation.loc[i]['filename'], annotation.loc[i]['class']
                # name = self.samples[img_name_to_idx[img_name]]
                targets[img_name] = classes_to_idx[class_id]


            self.targets = targets ### YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        ### YOUR CODE HERE
        name = self.samples[index]
        pth = self.root + '/' + name
        id = self.targets[name] if self.targets is not None else -1
        image = Image.open(pth).convert("RGB")
        image = np.array(image).astype(np.float32)
        return self.transform(image=image)['image'], name, id

    def __len__(self):
        return len(self.samples)


def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)


class ConvNext(torch.nn.Module):


    def __init__(self,internal_features = 1024, init_weights = False, unfreeze = 2):
        super().__init__()
        if init_weights:
            self.net = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.DEFAULT)
        else:
            self.net = torchvision.models.convnext_small()

        self.net.classifier[-1] = torch.nn.LazyLinear(internal_features)
        self.additional_layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(internal_features),
            torch.nn.GELU(),
            torch.nn.LazyLinear(CLASSES_CNT)
        )

        for child in list(self.net.children()):
            for param in child.parameters():
                param.requires_grad = True

        for child in list(self.net.children())[:-unfreeze]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(self.additional_layer(x), dim=1)

class CustomNetwork(pl.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.
    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """
    def __init__(self, features_criterion = None, internal_features = 1024, init_weights = False):
        super(CustomNetwork, self).__init__()
        ### YOUR CODE HERE
        self.features_criterion = features_criterion
        self.model = ConvNext(internal_features=internal_features, init_weights=init_weights, unfreeze=2)

        self.loss_f = F.nll_loss
        self.metric = lambda logits, y: torch.sum(logits.argmax(axis=1) == y)/ y.shape[0] # accuracy
        self.train_acc = []
        self.val_acc = []

        self.lr =  5e-4 #1e-3
        self.weight_decay = 1e-8
        self.best_val_acc = 0

    def training_step(self, batch, batch_idx):
        x, name, y_gt = batch
        y_pr = self.model(x)
        loss = self.loss_f(y_pr, y_gt) #y_gt.float()
        acc = self.metric(y_pr, y_gt)
        metrics = {'loss':loss.detach(), 'accuracy': acc.detach()}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.train_acc.append(acc.detach())  # optional

        return loss

    def validation_step(self, batch, batch_idx):
        x, name, y_gt = batch
        y_pr = self.model(x)
        loss = self.loss_f(y_pr, y_gt)  # .float()
        acc = self.metric(y_pr, y_gt)
        metrics = {'val_accuracy': acc.detach()}
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.val_acc.append(acc.detach())

    def on_validation_end(self):
        avg_acc = torch.stack(self.val_acc).mean()
        if avg_acc > self.best_val_acc:
            self.best_val_acc = avg_acc
            torch.save(self.model.state_dict(), 'best_current.pt')
        self.val_acc.clear()

    def configure_optimizers(self):
        params = list(self.model.named_parameters())
        grouped_parameters = [
            {"params": [p for _, p in params[:-3]], "lr": self.lr / 20},
            {"params": [p for _, p in params[-3:]], "lr": self.lr}
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=20,
                verbose=True,
                threshold=1e-2
        )
        lr_dict = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "loss",
        }
        return [optimizer], [lr_dict]

    def predict(self, x):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param x: батч с картинками
        """
        return self.net(x).argmax(axis = 1).detach().cpu().numpy()




def train_simple_classifier():
    """Функция для обучения простого классификатора на исходных данных."""
    ### YOUR CODE HERE

    root_folders = ['./cropped-train']
    classes_json = 'classes.json'
    batch_size = 16
    epochs_num = 5

    train_dataset = DatasetRTSD(root_folders, classes_json)

    valid_dataset = TestData('./smalltest', classes_json, 'smalltest_annotations.csv')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)

    train_module = CustomNetwork(init_weights=True)


    trainer = pl.Trainer(accelerator='cuda', devices = 1, max_epochs=epochs_num)

    trainer.fit(train_module, train_dataloaders=train_loader, val_dataloaders=valid_loader)



    return train_module


def apply_classifier(model, test_folder, path_to_classes_json):
    """
    Функция, которая применяет модель и получает её предсказания.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    results = ... ### YOUR CODE HERE - список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    return results


def test_classifier(model, test_folder, path_to_classes_json, annotations_file):
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    ### YOUR CODE HERE
    return total_acc, rare_recall, freq_recall


class SignGenerator(object):
    """
    Класс для генерации синтетических данных.
    :param background_path: путь до папки с изображениями фона
    """
    def __init__(self, background_path):
        ### YOUR CODE HERE
        pass

    def get_sample(self, icon):
        """
        Функция, встраивающая иконку на случайное изображение фона.
        :param icon: Массив с изображением иконки
        """
        icon = ... ### YOUR CODE HERE
        bg = ... ### YOUR CODE HERE - случайное изображение фона
        return ### YOUR CODE HERE


def generate_one_icon(args):
    """
    Функция, генерирующая синтетические данные для одного класса.
    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    ### YOUR CODE HERE


def generate_all_data(output_folder, icons_path, background_path, samples_per_class = 1000):
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.
    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    with ProcessPoolExecutor(8) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier():
    """Функция для обучения простого классификатора на смеси исходных и ситетических данных."""
    ### YOUR CODE HERE
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """
    def __init__(self, margin: float) -> None:
        super(FeaturesLoss, self).__init__()
        ### YOUR CODE HERE
        pass


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.
    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """
    def __init__(self, data_source, elems_per_class, classes_per_batch):
        ### YOUR CODE HERE
        pass
    def __iter__(self):
        ### YOUR CODE HERE
        pass


def train_better_model():
    """Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки."""
    ### YOUR CODE HERE
    return model


class ModelWithHead:
    """
    Класс, реализующий модель с головой из kNN.
    :param n_neighbors: Количество соседей в методе ближайших соседей
    """
    def __init__(self, n_neighbors):
        ### YOUR CODE HERE
        pass

    def load_nn(self, nn_weights_path):
        """
        Функция, загружающая веса обученной нейросети.
        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE
        pass

    def load_head(self, knn_path):
        """
        Функция, загружающая веса kNN (с помощью pickle).
        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE
        pass

    def predict(self, imgs):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param imgs: батч с картинками
        """
        features, model_pred = ... ### YOUR CODE HERE - предсказание нейросетевой модели
        features = features / np.linalg.norm(features, axis=1)[:, None]
        knn_pred = ... ### YOUR CODE HERE - предсказание kNN на features
        return knn_pred


class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    def __init__(self, data_source, examples_per_class) -> None:
        ### YOUR CODE HERE
        pass
    def __iter__(self):
        """Функция, которая будет генерировать список индексов элементов в батче."""
        return ### YOUR CODE HERE


def train_head(nn_weights_path, examples_per_class = 20):
    """
    Функция для обучения kNN-головы классификатора.
    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE
