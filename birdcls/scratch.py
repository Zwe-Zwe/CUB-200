"""CNNs trained from scratch (random initialisation, no pretrained weights).

These models form the "ablation ladder" (Track 1): start from a plain VGG-style network and add one
design idea at a time, measuring its effect on validation accuracy and on overfitting.

Model names
-----------
scratch_vgg      plain VGG-style CNN: (conv-ReLU-conv-ReLU-maxpool) x 5, then a large fully connected head
scratch_vgg_bn   the same network with BatchNorm after every convolution
scratch_vgg_gap  scratch_vgg_bn with the FC head replaced by global average pooling + one linear layer
scratch_resnet18 ResNet-18-style residual network (He et al., 2016), written here from scratch
"""
import torch.nn as nn

# Output channels of the five convolutional blocks. Each block halves the spatial size (max pooling),
# so a 224 x 224 image becomes a 7 x 7 x 256 feature map.
VGG_CHANNELS = (32, 64, 128, 256, 256)


def conv_block(in_c, out_c, batchnorm):
    """Two 3x3 convolutions (each followed by ReLU, optionally BatchNorm) and a 2x2 max pool.

    Two stacked 3x3 convolutions see a 5x5 region with fewer parameters than one 5x5 convolution and add
    an extra non-linearity (the VGG design idea). padding=1 keeps height and width unchanged; the max
    pool then halves them. With BatchNorm the convolution bias is redundant (BN has its own shift).
    """
    layers = []
    for c_in in (in_c, out_c):
        layers.append(nn.Conv2d(c_in, out_c, kernel_size=3, padding=1, bias=not batchnorm))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_c))      # normalise each channel over the batch
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(2))                    # H, W -> H/2, W/2
    return nn.Sequential(*layers)


class VGGNet(nn.Module):
    """VGG-style CNN with one of two classifier heads.

    features : 5 conv blocks, 3 -> 32 -> 64 -> 128 -> 256 -> 256 channels, 224 px -> 7 x 7

    head="fc"  : flatten (256 x 7 x 7 = 12,544 values) -> FC 512 -> ReLU -> dropout -> FC num_classes
                 The first FC layer alone has 12,544 x 512 = 6.4 M weights (about 70% of the model). With only
                 ~19 training images per class this head can memorise the training set.
    head="gap" : global average pooling (each of the 256 feature maps -> its mean, 1 number) -> dropout ->
                 FC num_classes. Only 256 x 200 + 200 = 51,400 head parameters (127x fewer). Averaging over
                 all positions also makes the prediction less dependent on *where* in the image a feature is,
                 and forces each channel to act as a detector for a pattern (Lin et al., "Network in Network").
    """

    def __init__(self, num_classes, batchnorm=False, channels=VGG_CHANNELS, fc_dim=512, dropout=0.0,
                 head="fc"):
        super().__init__()
        blocks, in_c = [], 3
        for out_c in channels:
            blocks.append(conv_block(in_c, out_c, batchnorm))
            in_c = out_c
        self.features = nn.Sequential(*blocks)
        if head == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)            # 256 x 7 x 7 -> 256 x 1 x 1 (global average)
            self.classifier = nn.Sequential(
                nn.Flatten(),                              # -> 256 values
                nn.Dropout(dropout),
                nn.Linear(in_c, num_classes),              # raw class scores (logits)
            )
            self._init_weights()
            return
        # Fixes the feature map at 7 x 7 whatever the input size (a no-op for 224 px input), so the
        # FC layer's input size does not depend on the image resolution.
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_c * 7 * 7, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),               # raw class scores (logits)
        )
        self._init_weights()

    def _init_weights(self):
        """He (Kaiming) initialisation keeps the activation variance roughly constant through ReLU layers,
        which a plain network without BatchNorm needs in order to train at all."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def get_classifier(self):
        """The final layer (same interface as timm models, used by models.split_params)."""
        return self.classifier[-1]

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class BasicBlock(nn.Module):
    """Residual block: out = ReLU( F(x) + shortcut(x) ), with F = conv3x3-BN-ReLU-conv3x3-BN.

    The block only has to learn the *change* F(x) to its input. Because the identity path carries the signal
    (and the gradient) straight through, very deep networks stay trainable, which plain stacks of
    convolutions (like the VGG model above) are not.
    When the block halves the resolution (stride 2) or changes the number of channels, x no longer has the
    shape of F(x), so the shortcut becomes a 1x1 convolution (stride 2) + BN that projects x to the new shape.
    """

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_c))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))      # the residual (skip) connection


class ResNet(nn.Module):
    """ResNet-18 layout for 224 x 224 input:

    stem    : conv7x7 (3 -> 64, stride 2) - BN - ReLU - maxpool 3x3 (stride 2)          224 -> 56
    stage 1 : 2 BasicBlocks,  64 channels                                               56 x 56
    stage 2 : 2 BasicBlocks, 128 channels (first block stride 2)                        28 x 28
    stage 3 : 2 BasicBlocks, 256 channels (first block stride 2)                        14 x 14
    stage 4 : 2 BasicBlocks, 512 channels (first block stride 2)                         7 x  7
    head    : global average pooling -> dropout -> linear 512 -> num_classes

    Compared with scratch_vgg_gap (2.4 M parameters, 10 conv layers, 256 final channels), this is deeper
    (17 conv layers + 3 shortcut projections), wider (512 final channels) and about 4.7x larger
    (11.3 M parameters), while the early downsampling stem keeps the compute similar.
    """

    def __init__(self, num_classes, blocks=(2, 2, 2, 2), widths=(64, 128, 256, 512), dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        layers, in_c = [], widths[0]
        for i, (n, w) in enumerate(zip(blocks, widths)):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1   # halve the resolution at the start of stages 2-4
                layers.append(BasicBlock(in_c, w, stride))
                in_c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_c, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        # "Zero-init residual" (Goyal et al., 2017): the last BN of every residual branch starts at gamma = 0, so
        # each block starts as an identity mapping. The network then behaves like a shallow one at the start
        # and grows into its depth, which makes early training more stable.
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def get_classifier(self):
        return self.classifier[-1]

    def forward(self, x):
        return self.classifier(self.pool(self.features(self.stem(x))))


SCRATCH_MODELS = {
    "scratch_vgg":    lambda n, dropout: VGGNet(n, batchnorm=False, dropout=dropout),
    "scratch_vgg_bn": lambda n, dropout: VGGNet(n, batchnorm=True, dropout=dropout),
    "scratch_vgg_gap": lambda n, dropout: VGGNet(n, batchnorm=True, dropout=dropout, head="gap"),
    "scratch_resnet18": lambda n, dropout: ResNet(n, dropout=dropout),
}


def is_scratch(model_name):
    return model_name in SCRATCH_MODELS


def build_scratch(model_name, num_classes, dropout=0.0):
    return SCRATCH_MODELS[model_name](num_classes, dropout)
