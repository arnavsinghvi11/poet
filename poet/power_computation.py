from typing import List

import numpy as np

# FLOPS_PER_WATT is FLOP_PER_JOULE

from poet.chipsets import MKR1000, JetsonTX2

device = MKR1000


class DNNLayer:
    _counter = 0

    def __init__(self, out_shape, depends_on: List["DNNLayer"] = tuple(), param_count=0):
        assert out_shape is not None  # get around varargs restriction
        self.unique_idx = "{}{:02d}".format(self.__class__.__name__, DNNLayer._counter)
        # self.unique_idx = "{}{:02d}".format(self.__class__.__name__, id(self) % 100)
        DNNLayer._counter += 1
        self.extra_repr_params = {}
        self.out_shape = out_shape
        self.depends_on = depends_on
        self.param_count = param_count

    def energy(self, device) -> float:
        """
        Energy consumed for computing the activations of the given layer
        @param device Choose a device - associated parameters
        @returns energy in joules
        """
        return self.flop / device["FLOPS_PER_WATT"]

    def runtime(self, device) -> float:
        """
        Time taken in (s) for computing the activations on CPU
        @param device Choose a device - associated parameters
        @returns energy in joules
        """
        return self.flop / device["FLOPS"]

    def energy_ram2sd(self, device) -> float:
        """
        Energy consumed for paging out activations of the given layer
        from SD card. Assuming we read 512 byte blocks.
         @param device Choose a device - associated parameters
         @returns energy in joules
        """
        _time = device["PAGEOUT_LATENCY"] + (self.output_ram_usage(device) / device["PAGEOUT_THROUGHPUT"])
        return _time * device["MEMORY_POWER"]

    def energy_sd2ram(self, device) -> float:
        """
        Energy consumed for paging out activations of the given layer
        from SD card. Assuming we read 512 byte blocks.
        @param device Choose a device - associated parameters
        @returns energy in joules
        """
        _time = device["PAGEIN_LATENCY"] + (self.output_ram_usage(device) / device["PAGEIN_THROUGHPUT"])
        return _time * device["MEMORY_POWER"]

    def output_ram_usage(self, device) -> int:
        """
        RAM consumption in bytes
        @returns memory required to store output of this layer
        """
        return np.prod(self.out_shape) * device["TYPE"]

    def param_ram_usage(self, device) -> int:
        """
        RAM necessary for parameters + workspace memory
        @returns memory required to store
        """
        return self.param_count * device["TYPE"]

    def __repr__(self):
        args = self.extra_repr_params
        args["out_shape"] = self.out_shape
        args["param_count"] = self.param_count
        args["depends_on"] = "[{}]".format(", ".join([x.unique_idx for x in self.depends_on]))
        return "{}({})".format(self.unique_idx, ",".join(["{}={}".format(k, v) for k, v in args.items()]))


class InputLayer(DNNLayer):
    def __init__(self, out_size):
        super().__init__(out_size, [])
        self.flop = 0


class LinearLayer(DNNLayer):
    def __init__(self, in_features: int, out_features: int, input: DNNLayer):
        super().__init__(
            self.find_outshape(in_features, out_features, input),
            [input] if input is not None else [],
            param_count=((in_features + 1) * out_features),
        )
        self.extra_repr_params["in_features"] = in_features
        self.extra_repr_params["out_features"] = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.flop = 2 * self.param_count + self.out_features

    def find_outshape(self, in_features, out_features, input):
        #come back to this assert statement
        # assert len(input.out_shape) == 2 and input.out_shape[1] == in_features, f"{input.out_shape}, {in_features}"
        return (input.out_shape[0], out_features)


class Conv2dLayer(DNNLayer):
    def __init__(self, in_features: int, out_channels: int, kernel_size, stride, padding, input: DNNLayer):
        """
        Differs from pytorch as pytorch param1 is in_channels, not in_features.
        Here we assume that the in_features is  Channels_In
        Kernel must always be [n X n]
        We assume bias
        """
        super().__init__(
            self.find_outshape(in_features, out_channels, kernel_size, stride, padding, input),
            [input] if input is not None else [],
            param_count=(out_channels * in_features * np.prod(kernel_size) + out_channels),
        )
        self.extra_repr_params["in_features"] = in_features
        self.extra_repr_params["out_features"] = out_channels
        self.flop = 2 * ((np.prod(kernel_size) * np.prod(input.out_shape) * out_channels)) + np.prod(self.out_shape)

    def find_outshape(self, in_features, out_channels, kernel_size, stride, padding, input):
        assert len(input.out_shape) == 4 and input.out_shape[1] == in_features, input.out_shape
        height = ((input.out_shape[2] - kernel_size[0] + 2 * padding[0]) // stride) + 1
        weight = ((input.out_shape[3] - kernel_size[1] + 2 * padding[1]) // stride) + 1
        return (input.out_shape[0], out_channels, height, weight)
    
class InPlaceConv2dLayer(DNNLayer):
    def __init__(self, in_features: int, out_channels: int, kernel_size, stride, padding, input: DNNLayer):
        """
        Differs from pytorch as pytorch param1 is in_channels, not in_features.
        Here we assume that the in_features is  Channels_In
        Kernel must always be [n X n]
        We assume bias
        """
        super().__init__(
            self.find_outshape(in_features, out_channels, kernel_size, stride, padding, input),
            [input] if input is not None else [],
            param_count=(out_channels * in_features * np.prod(kernel_size) + out_channels),
        )
        self.extra_repr_params["in_features"] = in_features
        self.extra_repr_params["out_features"] = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.flop = 2 * ((np.prod(kernel_size) * np.prod(input.out_shape) * out_channels)) + np.prod(self.out_shape)

    def find_outshape(self, in_features, out_channels, kernel_size, stride, padding, input):
        assert len(input.out_shape) == 4 and input.out_shape[1] == in_features, input.out_shape
        height = ((input.out_shape[2] - kernel_size[0] + 2 * padding[0]) // stride) + 1
        weight = ((input.out_shape[3] - kernel_size[1] + 2 * padding[1]) // stride) + 1
        return (input.out_shape[0], out_channels, height, weight)

    def forward(self, input: DNNLayer):
        out_channels = self.extra_repr_params["out_features"]
        padded_input = np.pad(input, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))
        batch_size, in_channels, height, width = input.shape
        output_height = (height - self.kernel_size[0] + 2 * self.padding[0]) // self.stride + 1
        output_width = (width - self.kernel_size[1] + 2 * self.padding[1]) // self.stride + 1
        output = np.zeros((batch_size, out_channels, output_height, output_width))

        for i in range(out_channels):
            for j in range(self.kernel_size[0]):
                for k in range(self.kernel_size[1]):
                    output_channel = padded_input[:, :, j : j+output_height*self.stride : self.stride, k : k+output_width*self.stride : self.stride]
                    output[:, i:i+1, :, :] += output_channel
        return output

class Conv2dPatchedLayer(DNNLayer):
    def __init__(self, in_features: int, out_channels: int, kernel_size: tuple, stride: tuple, padding: tuple, patch_size: int, input_layer: DNNLayer):
        """
        Differs from pytorch as pytorch param1 is in_channels, not in_features.
        Here we assume that the in_features is  Channels_In
        Kernel must always be [n X n]
        We assume bias
        """
        super().__init__(
            self.find_outshape(in_features, out_channels, kernel_size, stride, padding, patch_size, input_layer),
            [input_layer] if input_layer is not None else [],
            param_count=(out_channels * in_features * np.prod(kernel_size) + out_channels),
        )
        self.extra_repr_params["in_features"] = in_features
        self.extra_repr_params["out_features"] = out_channels
        self.flop = 2 * ((np.prod(kernel_size) * np.prod(input_layer.out_shape) * out_channels)) + np.prod(self.out_shape)
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def find_outshape(self, in_features, out_channels, kernel_size, stride, padding, patch_size, input_layer):
        #come back to this # assert len(input_layer.out_shape) == 4 and input_layer.out_shape[1] == in_features, input_layer.out_shape
        height = ((input_layer.out_shape[2] - kernel_size[0] + 2 * padding[0]) // stride) + 1
        width = ((input_layer.out_shape[3] - kernel_size[1] + 2 * padding[1]) // stride) + 1
        output_height = ((height - kernel_size[0] + 2 * padding[0]) // patch_size[0]) + 1
        output_width = ((width - kernel_size[1] + 2 * padding[1]) // patch_size[1]) + 1
        return (input_layer.out_shape[0], out_channels, output_height, output_width)
    
    def conv2d_patch(self, patch, kernel_size):
        height = patch.shape[2] - kernel_size[0] + 1
        width = patch.shape[3] - kernel_size[1] + 1
        output = np.zeros((patch.shape[0], self.out_shape[2], self.out_shape[3]))
        for k in range(patch.shape[0]):
            for i in range(height):
                for j in range(width):
                    output[k, i, j] = np.sum(patch[k, :, i:i+kernel_size[0], j:j+kernel_size[1]])
        return output

    def forward(self, input: DNNLayer):
        patch_height = self.patch_size
        patch_width = self.patch_size
        output = np.zeros((input.shape[0], self.out_shape[2], self.out_shape[3], self.out_shape[1]))

        # Loop over patches and perform patch-wise convolution
        for i in range(0, input.shape[2], patch_height):
            for j in range(0, input.shape[3], patch_width):
                patch = input[:, :, i:i+patch_height, j:j+patch_width]
                patch_output = self.conv2d_patch(patch, self.kernel_size)
                output[:, i:i+patch_output.shape[0], j:j+patch_output.shape[1], :] = patch_output

        return output

class BatchNorm2d(DNNLayer):
    """
    Here we assume the input to be C in [N X C X H X W]
    The input and output shape are same as preceding layer.
    We do not assume batchnorm is folded in conv layers.
    """

    def __init__(self, input: DNNLayer):
        super().__init__(input.out_shape, [input] if input is not None else [], param_count=(4 * input.out_shape[1]))
        self.extra_repr_params["in_features"] = input.out_shape
        self.extra_repr_params["out_features"] = input.out_shape
        self.in_features = input.out_shape
        self.out_features = input.out_shape
        # This is flawed. But, so is the world [https://github.com/clavichord93/CNN-Calculator/blob/master/CNNCalculator.py#L83]
        self.flop = 2 * np.prod(self.in_features)


class ReLULayer(DNNLayer):
    def __init__(self, input: DNNLayer):
        super().__init__(out_shape=input.out_shape, depends_on=[input], param_count=0)
        self.flop = np.prod(self.out_shape)


class MaxPool2d(DNNLayer):
    def __init__(self, kernel_size, stride, input: DNNLayer):
        super().__init__(out_shape=self.find_outshape(kernel_size, stride, input), depends_on=[input], param_count=0)
        # Flops assume stride=1 for legacy reasons
        self.flop = 2 * (np.prod(kernel_size) * np.prod(self.out_shape))

    def find_outshape(self, kernel_size, stride, input):
        B, C, H, W = input.out_shape
        return (B, C, ((H - kernel_size[0]) // stride) + 1, ((W - kernel_size[1]) // stride) + 1)


class AvgPool2d(DNNLayer):
    # Assumed to be same as MaxPool2d
    def __init__(self, kernel_size, stride, input: DNNLayer):
        super().__init__(out_shape=self.find_outshape(kernel_size, stride, input), depends_on=[input], param_count=0)
        # Flops assume stride=1 for legacy reasons
        self.flop = 2 * (np.prod(kernel_size) * np.prod(self.out_shape))

    def find_outshape(self, kernel_size, stride, input):
        B, C, H, W = input.out_shape
        return (B, C, (H - kernel_size[0]) // stride + 1, (W - kernel_size[1]) // stride + 1)


class GlobalAvgPool(DNNLayer):
    # Average across channel for [H X W] layers
    def __init__(self, input: DNNLayer):
        super().__init__(out_shape=self.find_outshape(input), depends_on=[input], param_count=0)
        self.flop = 2 * np.prod(input.out_shape)

    def find_outshape(self, input):
        B, C, H, W = input.out_shape
        return (B, C)


class TanHLayer(DNNLayer):
    def __init__(self, input: DNNLayer):
        super().__init__(out_shape=input.out_shape, depends_on=[input], param_count=0)
        # Assuming O(100) flops per tanH
        self.flop = 100 * np.prod(self.out_shape)


class SigmoidLayer(DNNLayer):
    def __init__(self, input: DNNLayer):
        super().__init__(out_shape=input.out_shape, depends_on=[input], param_count=0)
        # Assuming one flop each for each distinct operation.
        # Exponentiation is assumed as 1 flop !?
        self.flop = 4 * np.prod(self.out_shape)


class SkipAddLayer(DNNLayer):
    """
    This is especially for the short-circuit connection that is needed for us to generate dependencies
    """

    def __init__(self, input1: DNNLayer, input2: DNNLayer):
        super().__init__(out_shape=input1.out_shape, depends_on=[input1, input2], param_count=0)
        self.flop = np.prod(self.out_shape)


class FlattenLayer(DNNLayer):
    def __init__(self, input: DNNLayer):
        super().__init__(out_shape=(input.out_shape[0], np.prod(input.out_shape[1:])), depends_on=[input], param_count=0)
        self.flop = np.prod(self.out_shape)


class DropoutLayer(DNNLayer):
    def __init__(self, input: DNNLayer):
        super().__init__(out_shape=input.out_shape, depends_on=[input], param_count=0)
        self.flop = np.prod(self.out_shape)


class CrossEntropyLoss(DNNLayer):
    def __init__(self, input: DNNLayer, n_classes=10):
        super().__init__(out_shape=(n_classes,), depends_on=[input])
        self.extra_repr_params["n_classes"] = n_classes
        self.n_classes = n_classes
        self.flop = np.prod(self.depends_on[0].out_shape) * self.n_classes


class GradientLayer(DNNLayer):
    def __init__(self, output: DNNLayer, inputs: List[DNNLayer], grad_outputs: "GradientLayer"):
        super().__init__(out_shape=self.find_outshape(inputs), depends_on=[output, *inputs, grad_outputs])
        self.flop = 2 * output.flop

    def find_outshape(self, inputs):
        for _index in range(1, len(inputs)):
            assert inputs[_index].out_shape == inputs[_index - 1].out_shape
        return inputs[0].out_shape

#common operator fusions:

class Conv2dBNLayer(DNNLayer):
    def __init__(self, in_features: int, out_channels: int, kernel_size, stride, padding, input: DNNLayer):
        """
        Differs from pytorch as pytorch param1 is in_channels, not in_features.
        Here we assume that the in_features is  Channels_In
        Kernel must always be [n X n]
        We assume bias
        """
        super().__init__(
            self.find_outshape(in_features, out_channels, kernel_size, stride, padding, input),
            [input] if input is not None else [],
            param_count=(out_channels * in_features * np.prod(kernel_size) + out_channels),
        )
        self.extra_repr_params["in_features"] = in_features
        self.extra_repr_params["out_features"] = out_channels
        self.flop = 2 * ((np.prod(kernel_size) * np.prod(input.out_shape) * out_channels)) + np.prod(self.out_shape)
        self.conv = Conv2dLayer(in_features, out_channels, kernel_size, stride, padding, input)
        self.bn = BatchNorm2d(input)

    def find_outshape(self, in_features, out_channels, kernel_size, stride, padding, input):
        assert len(input.out_shape) == 4 and input.out_shape[1] == in_features, input.out_shape
        height = ((input.out_shape[2] - kernel_size[0] + 2 * padding[0]) // stride) + 1
        weight = ((input.out_shape[3] - kernel_size[1] + 2 * padding[1]) // stride) + 1
        return (input.out_shape[0], out_channels, height, weight)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class Conv2dReLULayer(DNNLayer):
    def __init__(self, in_features: int, out_channels: int, kernel_size, stride, padding, input: DNNLayer):
        """
        Differs from pytorch as pytorch param1 is in_channels, not in_features.
        Here we assume that the in_features is  Channels_In
        Kernel must always be [n X n]
        We assume bias
        """
        super().__init__(
            self.find_outshape(in_features, out_channels, kernel_size, stride, padding, input),
            [input] if input is not None else [],
            param_count=(out_channels * in_features * np.prod(kernel_size) + out_channels),
        )
        self.extra_repr_params["in_features"] = in_features
        self.extra_repr_params["out_features"] = out_channels
        self.flop = 2 * ((np.prod(kernel_size) * np.prod(input.out_shape) * out_channels)) + np.prod(self.out_shape)
        self.conv = Conv2dLayer(in_features, out_channels, kernel_size, stride, padding, input)
        self.relu = ReLULayer(input)

    def find_outshape(self, in_features, out_channels, kernel_size, stride, padding, input):
        assert len(input.out_shape) == 4 and input.out_shape[1] == in_features, input.out_shape
        height = ((input.out_shape[2] - kernel_size[0] + 2 * padding[0]) // stride) + 1
        weight = ((input.out_shape[3] - kernel_size[1] + 2 * padding[1]) // stride) + 1
        return (input.out_shape[0], out_channels, height, weight)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

#detect the 2 layers being fused
#Fused2Layers(Layer1(params*), Layer2(params*))

class Fused2Layers(DNNLayer):
    def __init__(self, layer1: DNNLayer, layer2: DNNLayer):
        super().__init__(
            self.find_outshape(layer1, layer2),
            [layer1, layer2],
            param_count=layer1.param_count + layer2.param_count,
        )
        self.flop = layer1.flop + layer2.flop
        self.layer1 = layer1
        self.layer2 = layer2
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
class Fused3Layers(DNNLayer):
    def __init__(self, layer1: DNNLayer, layer2: DNNLayer, layer3: DNNLayer):
        super().__init__(
            self.find_outshape(layer1, layer2, layer3),
            [layer1, layer2, layer3],
            param_count=layer1.param_count + layer2.param_count + layer3.param_count,
        )
        self.flop = layer1.flop + layer2.flop + layer3.flop
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
def get_net_costs(net, device):
    compute_energy_list, compute_runtime_list, ram_list, param_ram_list, pagein_cost, pageout_cost = [[] for _ in range(6)]
    """ 
    If you have access to hardware device, this costs should be obtained from accurate profiles. 
    Else, POET adopts a flop based model.
    """
    for layer in net:
        compute_energy_list.append(layer.energy(device))
        compute_runtime_list.append(layer.runtime(device))
        ram_list.append(layer.output_ram_usage(device))
        param_ram_list.append(layer.param_ram_usage(device))
        pagein_cost.append(layer.energy_sd2ram(device))
        pageout_cost.append(layer.energy_ram2sd(device))

    return dict(
        runtime_ms=compute_runtime_list,
        memory_bytes=ram_list,
        param_memory_bytes=param_ram_list,
        compute_cost_joules=compute_energy_list,
        pagein_cost_joules=pagein_cost,
        pageout_cost_joules=pageout_cost,
    )
