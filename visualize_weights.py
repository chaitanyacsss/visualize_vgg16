import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
from torch import nn
from torchvision import utils

RESULTS_FOLDER_WEIGHTS = os.path.join("results", "weights")
if not os.path.exists(RESULTS_FOLDER_WEIGHTS):
    os.makedirs(RESULTS_FOLDER_WEIGHTS)


def vis_tensor(tensor, ch=0, allkernels=False, nrow=8, padding=1, layer_number=0):
    """
    vis_tensor: visuzlization tensor
        @ch: visualization channel
        @allkernels: visualization all tensores
    """

    n, c, w, h = tensor.shape

    np.save(os.path.join(RESULTS_FOLDER_WEIGHTS, "layer_" + str(layer_number) + "_weights.npy"),
            tensor.numpy())
    if allkernels and c != 3:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    if c == 3:
        nrow = 8
    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    print("shape of tensor before sending to grid: ", tensor.shape)
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()
    # plt.show()
    plt.savefig(os.path.join(RESULTS_FOLDER_WEIGHTS, "layer_" + str(layer_number) + "_weights.svg"))


if __name__ == '__main__':
    vgg = models.vgg16(pretrained=True)
    mm = vgg.double()
    count = 0
    layer_num = 0
    while count < 5:
        layer = mm.features[layer_num]
        if isinstance(layer, nn.Conv2d):
            weights_tensor = layer.weight.data.clone()
            count += 1
            print("tensor shape : ", weights_tensor.shape)
            vis_tensor(weights_tensor, ch=0, allkernels=True, nrow=weights_tensor.shape[1], layer_number=count)
        layer_num += 1
