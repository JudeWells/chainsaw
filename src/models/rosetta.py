from torch import nn, transpose


def elu():
    return nn.ELU(inplace=True)


def instance_norm(filters, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(filters, affine=True, eps=eps, **kwargs)


def conv2d(in_chan, out_chan, kernel_size, dilation=1, **kwargs):
    padding = dilation * (kernel_size - 1) // 2
    return nn.Conv2d(in_chan, out_chan, kernel_size, padding=padding, dilation=dilation, **kwargs)


class trRosettaNetwork(nn.Module):
    """symmetrise_output: if True, output is symmetrised by adding the transpose
    of the output and dividing by 2"""
    def __init__(self, filters=64, kernel=3, num_layers=61, in_channels=3, symmetrise_output=False):
        super().__init__()
        self.filters = filters
        self.kernel = kernel
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.symmetrise_output = symmetrise_output
        self.first_block = nn.Sequential(
            conv2d(self.in_channels, filters, 1),
            instance_norm(filters),
            elu()
        )
        self.output_layer = nn.Sequential(
            conv2d(filters, 1, kernel, dilation=1),
            nn.Sigmoid())


        # stack of residual blocks with dilations
        cycle_dilations = [1, 2, 4, 8, 16]
        dilations = [cycle_dilations[i % len(cycle_dilations)] for i in range(num_layers)]

        self.layers = nn.ModuleList([nn.Sequential(
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters),
            elu(),
            nn.Dropout(p=0.15),
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters)
        ) for dilation in dilations])

        self.activate = elu()


    def forward(self, x):
        x = self.first_block(x)

        for layer in self.layers:
            x = self.activate(x + layer(x))
        y_hat = self.output_layer(x)
        if self.symmetrise_output:
            return (y_hat + transpose(y_hat, -1, -2)) * 0.5
        else:
            return y_hat
