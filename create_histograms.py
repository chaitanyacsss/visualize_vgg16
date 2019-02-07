import os

import cv2
import joypy
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt

from misc_functions import convert_to_grayscale


def image_to_df(im_path, file_name):
    read_image = cv2.imread(im_path)
    print("Shape of the image file: ", read_image.shape)

    read_image = np.sum(np.abs(read_image), axis=3)
    im_max = np.percentile(read_image, 99)
    im_min = np.min(read_image)
    read_image = (np.clip((read_image - im_min) / (im_max - im_min), 0, 1))
    read_image = np.expand_dims(read_image, axis=2)

    read_image = convert_to_grayscale(read_image).ravel()
    image_df = pd.DataFrame(read_image, columns=[file_name.split(".jpg")[0]])
    return image_df


def npy_to_df(npy_path, file_name):
    read_npy = np.load(npy_path)
    print("Shape of the npy file: ", read_npy.shape)

    im_max = np.percentile(read_npy, 99)
    im_min = np.min(read_npy)
    read_npy = (np.clip((read_npy - im_min) / (im_max - im_min), 0, 1))

    read_npy = read_npy.ravel()
    print("Shape after ravel: ", read_npy.shape)
    npy_df = pd.DataFrame(read_npy, columns=[file_name.split("layer")[1].split("_")[0]])
    return npy_df


def get_layerwise_filter_histograms(input_read_folder, filter_number):
    all_frames = {}
    for file in os.listdir(input_read_folder):
        if file.endswith("filter" + str(filter_number) + "_Guided_BP_color.npy"):
            created_df = npy_to_df(os.path.join(input_read_folder, file), file)
            all_frames[file] = created_df

    full_result = pd.concat(all_frames.values(), axis=1)
    labels = [x for x in np.arange(0.0, 1.0, 0.01)]
    print(labels)
    # print(full_result)
    joypy.joyplot(full_result, labels=True, range_style='all',
                  grid="y", linewidth=1, legend=True, figsize=(10, 3),
                  title="Layer-wise activation/feature map at " + str(filter_number) + "th filter",
                  colormap=cm.autumn_r, fade=True)
    # , hist="True", bins=50
    plt.savefig(os.path.join("results", "layerwise_activations_filter_" + str(filter_number) + ".jpg"))


def weights_npy_to_df(npy_path, file_name):
    read_npy = np.load(npy_path)
    print("Shape of the npy file: ", read_npy.shape)

    im_max = np.percentile(read_npy, 99)
    im_min = np.min(read_npy)
    read_npy = (np.clip((read_npy - im_min) / (im_max - im_min), 0, 1))
    all_frames = {}
    for channel_num in range(read_npy.shape[1]):
        curr_channel = read_npy[:, channel_num, :, :]
        curr_channel = curr_channel.ravel()
        curr_df = pd.DataFrame(curr_channel, columns=[channel_num])
        all_frames[channel_num] = curr_df
    full_result = pd.concat(all_frames.values(), axis=1)
    return full_result


def get_weights_histograms(input_read_folder, layer_num):
    try:
        full_result = None
        for file in os.listdir(input_read_folder):
            if file.endswith("layer_" + str(layer_num) + "_weights.npy"):
                full_result = weights_npy_to_df(os.path.join(input_read_folder, file), file)

        y_labels = (True if len(full_result.columns) < 50 else False)
        joypy.joyplot(full_result, labels=True, range_style='all',
                      grid="y", linewidth=1, legend=True, figsize=(6, 5), ylabels=y_labels,
                      title="Channel wise weights at " + str(layer_num) + "th layer",
                      colormap=cm.autumn_r, fade=True, bins=200)
        plt.savefig(os.path.join("results", "channelwise_weights_" + str(layer_num) + ".jpg"))
    except Exception as e:
        print("Error creating histograms for layer " + str(layer_num))
        print(e)


if __name__ == '__main__':
    filter_num = 10
    get_layerwise_filter_histograms(os.path.join("results", "layer_filter"), filter_num)

    for layer in range(1, 6):
        get_weights_histograms(os.path.join("results", "weights"), layer)
