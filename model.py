import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.0)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class LutGenerator(nn.Module):
    def __init__(self, n_channels, kernel_size=5, lut_points=32, use_batchnorm=True):
        super(LutGenerator, self).__init__()
        self.lut_points = lut_points
        model = []
        in_channels = n_channels
        out_channels = n_channels * 2
        for i in range(5):
            new_stack = [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=(2, 2))
            ]
            if use_batchnorm:
                new_stack = new_stack + [
                    nn.BatchNorm2d(out_channels)
                ]
            new_stack = new_stack + [
                nn.LeakyReLU(0.2)
            ]
            model = model + new_stack
            in_channels = out_channels
            out_channels = in_channels * 2
        model = model + [nn.Flatten()]
        model_tail = [nn.Linear(8192 * n_channels, 512)]
        model_tail = model_tail + [nn.Sigmoid()]
        model_tail = model_tail + [nn.Linear(512, lut_points * 3)]
        model_tail = model_tail + [nn.Sigmoid()]
        self.model = nn.Sequential(*model)
        self.model_tail = nn.Sequential(*model_tail)
        self.apply(weights_init)

    def forward(self, input_batch):
        lut_flat = self.model(input_batch)
        lut_flat = self.model_tail(lut_flat)
        lut = lut_flat.view(-1, 3, self.lut_points) * 255.0
        return lut

class LutGeneratorHist(nn.Module):
    def __init__(self, n_channels, lut_points=32):
        super(LutGeneratorHist, self).__init__()
        self.lut_points = lut_points
        model = [nn.Flatten()]
        model = model + [nn.Linear(256 * n_channels, 512)]
        model = model + [nn.Sigmoid()]
        model = model + [nn.Linear(512, lut_points * 3)]
        model = model + [nn.Sigmoid()]
        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, input_batch):
        lut_flat = self.model(input_batch)
        lut = lut_flat.view(-1, 3, self.lut_points) * 255.0
        return lut
