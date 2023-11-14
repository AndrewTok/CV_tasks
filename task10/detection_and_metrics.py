# ============================== 1 Classifier model ============================
import torch.utils.data
import torchvision.datasets.inaturalist


def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    from torch.nn import Sequential, Conv2d, LazyLinear, BatchNorm2d, ReLU, Flatten, Softmax, MaxPool2d

    h, w, c = input_shape

    model = Sequential(
        Conv2d(in_channels=c, out_channels=32, kernel_size=3),
        BatchNorm2d(32),
        ReLU(),
        MaxPool2d(kernel_size=2),

        Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        BatchNorm2d(64),
        ReLU(),
        MaxPool2d(kernel_size=2),

        Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        BatchNorm2d(128),
        ReLU(),

        Flatten(),
        LazyLinear(2),
        Softmax(dim=1)
    )




    return model
    # your code here /\

def fit_cls_model(X, y):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model((40, 100, 1))
    # train model
    import albumentations as A
    import pytorch_lightning as pl
    from albumentations.pytorch import ToTensorV2
    from torch.nn import functional as F
    from sklearn.metrics import accuracy_score
    from torch import from_numpy
    from sklearn.model_selection import train_test_split

    train_transform = A.Compose(
        [
            A.Rotate(limit=70, p=0.3),
            A.HorizontalFlip(p = 0.3),
            # A.Normalize(),
            ToTensorV2()
        ]
    )

    # test_transform = A.Compose(
    #     [
    #         A.Normalize(),
    #         A.pytorch.ToTensorV2()
    #     ]
    # )


    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, transforms):
            super().__init__()
            self.X = X
            self.y = y
            self.transforms = transforms


        def __getitem__(self, index):
            if self.transforms is not None:
                x = self.transforms(image = self.X[index])['image']
            else:
                x = self.X[index]
            return x, self.y[index]

        def __len__(self):
            return self.X.shape[0]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_test = from_numpy(X_train), from_numpy(X_test)
    # y_train, y_test = from_numpy(y_train), from_numpy(y_test)


    train_set = SimpleDataset(X, y, transforms=None)
    # test_set = SimpleDataset(X_test, y_test, test_transform)

    epochs_num = 2
    batch_size = 16



    # model.train()

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, drop_last=True)
    # valid_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, drop_last=True)

    class TrainModule(pl.LightningModule):

        def __init__(self, model):
            super().__init__()
            self.model = model
            self.lr = 1e-3
            self.weight_decay = 1e-8
            self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.loss_f = F.nll_loss
            self.train_acc = []
            self.metric = lambda logits, y: torch.sum(logits.argmax(axis=1) == y) / y.shape[0]  # accuracy

        def training_step(self, batch, batch_idx):
            x, y_gt = batch
            y_pr = self.model(x)

            loss = self.loss_f(y_pr, y_gt)
            self.train_acc.append(self.metric(y_pr.detach(), y_gt))

            return loss

        def on_train_epoch_end(self):
            ## display average loss across epoch
            avg_loss = torch.stack(self.train_acc).mean()
            print(
                f"Epoch {self.trainer.current_epoch}, "
                f"Train_acc: {round(float(avg_loss), 3)}",
            )
            # don't forget to clear the saved losses
            self.train_acc.clear()

        def configure_optimizers(self):
            """Define optimizers and LR schedulers."""
            return self.optimizer


    module = TrainModule(model)
    trainer = pl.Trainer(accelerator='cpu', max_epochs=epochs_num, devices=1,\
                         logger=False, enable_checkpointing=False)


    trainer.fit(module, train_loader)

    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    from torch.nn import Sequential, Conv2d, Sigmoid

    final_kernal_size = (6, 21)
    detection_model = Sequential(
        cls_model[0], #conv
        cls_model[1], #bn
        cls_model[2], #relu
        cls_model[3], #max pool

        cls_model[4], #conv
        cls_model[5], #bn
        cls_model[6], #relu
        cls_model[7], #max pool

        cls_model[8], #conv
        cls_model[9], #bn
        cls_model[10], #relu

        Conv2d(in_channels=cls_model[8].out_channels, out_channels=2, kernel_size=final_kernal_size),
        Sigmoid()
    )

    detection_model[-2].weight = cls_model[-2].weight.reshape(2, 128, final_kernal_size[0], final_kernal_size[1])
    detection_model[-2].bias = cls_model[-2].bias


    return detection_model
    # your code here /\


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    import numpy as np

    def receptive_bb(out_x, out_y):
        start_x = out_x*4
        end_x = np.cilp(start_x + 40, 0, 220)

        start_y = out_y*4
        end_y = np.clip(start_y + 100, 0, 370)

        n_rows = end_x - start_x
        n_cols = end_y - start_y
        return start_x, start_y, n_rows, n_cols #start_x, end_x, start_y, end_y

    detection_model.eval()
    detections = {}


    for filename in dictionary_of_images.key():
        img = dictionary_of_images[filename]
        pad_x = 220 - img.shape[0]
        pad_y = 370 - img.shape[1]
        padded = np.pad(img, ((0, pad_x), (0, pad_y)))
        pred = detection_model(padded[None, ...])
        flatten_pred = torch.flatten(pred, start_dim=2, end_dim=3)
        indices = np.arange(flatten_pred.shape[-1])

        detected_mask = np.argmax(flatten_pred.detach().numpy(), axis = 1).flatten()
        detected_ids = indices[detected_mask == 1]
        confidences = flatten_pred.detach().numpy()[detected_mask == 1][:, 1, :]

        out_x = indices//100
        out_y = indices - out_x*100
        rows, cols, n_rows, n_cols = receptive_bb(out_x, out_y)

        detects = np.array(5*len(detected_ids))
        detects[0::5] = rows
        detects[1::5] = cols
        detects[2::5] = n_rows
        detects[3::5] = n_cols
        detects[4::5] = confidences

        detections[filename] = detects




    return detections
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    return 1
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    return 1
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    return {}
    # your code here /\
