import os

import numpy as np
import torch
from config import VOC_CLASSES, bbox_util, model
from PIL import Image
from skimage import io
from skimage.transform import resize
from utils import get_color
# from skimage.io import imread

import albumentations as A
from albumentations.pytorch import ToTensorV2


def detection_cast(detections):
    """Helper to cast any array to detections numpy array.
    Even empty.
    """
    return np.array(detections, dtype=np.int32).reshape((-1, 5))


def rectangle(shape, ll, rr, line_width=5):
    """Draw rectangle on numpy array.

    rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
    frame[rr, cc] = [0, 255, 0] # Draw green bbox
    """
    ll = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(ll, 0))
    rr = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(rr, 0))
    result = []

    for c in range(line_width):
        for i in range(ll[0] + c, rr[0] - c + 1):
            result.append((i, ll[1] + c))
            result.append((i, rr[1] - c))
        for j in range(ll[1] + c + 1, rr[1] - c):
            result.append((ll[0] + c, j))
            result.append((rr[0] - c, j))

    return tuple(zip(*result))


IMAGENET_MEAN = np.array([103.939, 116.779, 123.68]).reshape(1, 1, 3)


def image2tensor(image):
    # Write code here
    image = image.astype('float32') # convert frame to float
    image -= IMAGENET_MEAN
    image = image[:, :, ::-1]# convert RGB to BGR

    h,w, c = image.shape
    transform = A.Compose(
        [
            A.Resize(300, 300),# resize image to 300x300
            ToTensorV2()# torch works with CxHxW images
        ]
    )
    # tensor.shape == (1, channels, height, width)
    tensor = transform(image=image)['image'].unsqueeze(0)
    return tensor, h, w


@torch.no_grad()
def extract_detections(frame, min_confidence=0.6, labels=None):
    """Extract detections from frame.

    frame: numpy array WxHx3
    returns: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
    """
    # Write code here
    # First, convert the input image to tensor
    input_tensor, h, w = image2tensor(frame)

    # Then use model(input_tensor),
    # convert output to numpy
    # and bbox_util.detection_out
    results = model(input_tensor).detach().numpy()
    results = np.array(bbox_util.detection_out(results, confidence_threshold=min_confidence))

    # Select detections with confidence > min_confidence
    # hint: see confidence_threshold argument of bbox_util.
    # If label set is known, use it
    if labels is not None:
        result_labels = results[:, 0].astype(np.int32)
        indices = [
            index
            for index, label in enumerate(result_labels)
            if VOC_CLASSES[label - 1] in labels
        ]
        results = results[indices]

    # Remove confidence column from result
    if results.shape[-1] != 0:
        results = np.delete(results, 1, axis=-1)
        # Resize detection coords to the original image shape.
        # Return result
        results[...,1] *= w
        results[..., 3] *= w
        results[...,2] *= h
        results[..., 4] *= h
    return detection_cast(results)


def draw_detections(frame, detections):
    """Draw detections on frame.

    Hint: help(rectangle) would help you.
    Use get_color(label) to select color for detection.
    """

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    frame = frame.copy()

    det_count = detections.shape[0]
    fig, ax = plt.subplots()
    # print("NEW FRAME")
    for i in range(det_count):
        bb = detections[i]
        w = bb[3] - bb[1]
        h = bb[4] - bb[2]
        x = bb[1]
        y = bb[2]
        rect = patches.Rectangle((y, x), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        rr, cc = rectangle(frame.shape, (bb[2], bb[1]), (bb[4], bb[3]))
        frame[rr, cc] = [0, 255, 0]  # Draw green bbox
        # print('label ', bb[0])
        # rect = patches.Rectangle((gt_bb[1], gt_bb[0]), gt_bb[3], gt_bb[2], linewidth=1, edgecolor='g', facecolor='none')
        # ax.add_patch(rect)

    ax.imshow(frame, cmap='gray')

    return frame


def main():
    dirname = os.path.dirname(__file__)
    frame = io.imread(os.path.join(dirname, "data", "new_york.jpg")) #Image.open(os.path.join(dirname, "data", "test.png"))
    frame = np.array(frame)

    detections = extract_detections(frame)
    frame = draw_detections(frame, detections)

    io.imshow(frame)
    io.show()


if __name__ == "__main__":
    main()
