import copy
import torch
from . import layers

def quantize_model(model, scale):
    # deferred import to prevent circular import
    # from . import layers

    quantized_model = copy.deepcopy(model)

    layer_mapping = {
        torch.nn.Linear: layers.QLinear,
        torch.nn.Conv2d: layers.QConv2d,
        torch.nn.Conv1d: layers.QConv1d,
        # torch.nn.BatchNorm2d: layers.QuantisedBatchNorm2d,
    }

    def replace_layers(module):
        for name, child in list(module.named_children()):
            if type(child) in layer_mapping:
                quantized_layer = layer_mapping[type(child)](child, scale)
                quantized_layer.qname += f"-{name}-{replace_layers.qid_counter}"
                setattr(module, name, quantized_layer)

                replace_layers.qid_counter += 1
            elif type(child) == torch.nn.ReLU:
                setattr(module, name, torch.nn.ReLU6())
            else:
                replace_layers(child)

    replace_layers.qid_counter = 0
    replace_layers(quantized_model)
    
    return quantized_model