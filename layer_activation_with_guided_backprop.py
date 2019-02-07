"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""

import torch
from tensorboardX import SummaryWriter
from torch.nn import ReLU

from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()
        self.writer = SummaryWriter()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class, cnn_layer, filter_pos):
        self.model.zero_grad()
        # Forward pass
        x = input_image
        for index, layer in enumerate(self.model.features):
            # Forward pass layer by layer
            # x is not used after this point because it is only needed to trigger
            # the forward hook function
            x = layer(x)
            # Only need to forward until the selected layer is reached
            if index == cnn_layer:
                # (forward hook function triggered)
                break
        conv_output = torch.sum(torch.abs(x[0, filter_pos]))
        # Backward pass
        conv_output.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        # self.writer.add_histogram("hist", gradients_as_arr, cnn_layer)
        return gradients_as_arr


if __name__ == '__main__':
    filter_pos = 10
    target_example = 0  # snake
    (original_image, prep_img, target_class, file_name_to_export_base, pretrained_model) = \
        get_example_params(target_example)

    for cnn_layer in range(0, 5):
        file_name_to_export = file_name_to_export_base + '_layer' + str(cnn_layer) + '_filter' + str(filter_pos)
        # Guided backprop
        # file_name_to_export = os.path.join("layer_filter", file_name_to_export)
        GBP = GuidedBackprop(pretrained_model)
        # Get gradients
        guided_grads = GBP.generate_gradients(prep_img, target_class, cnn_layer, filter_pos)

        save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
        # Convert to grayscale
        grayscale_guided_grads = convert_to_grayscale(guided_grads)
        # Save grayscale gradients

        save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
        # Positive and negative salience maps
        pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
        save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
        save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
        print('Layer Guided backprop completed for Convolution layer ', str(cnn_layer + 1))
