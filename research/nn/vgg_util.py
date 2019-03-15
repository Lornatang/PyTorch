import torch.nn as nn
import math

NUM_CLASSES = 10


class VGG(nn.Module):
  
  def __init__(self, features, num_classes=NUM_CLASSES, init_weights=True):
    super(VGG, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(
      nn.Linear(512 * 1 * 1, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(4096, num_classes),
    )
    if init_weights:
      self._initialize_weights()
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
  
  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


def make_layers(config, batch_norm=False):
  layers = []
  in_channels = 3
  for v in config:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)


cfg = {
  'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
  """VGG 11-layer model (configuration "A")
  """
  model = VGG(make_layers(cfg['A']), **kwargs)
  return model


def vgg11_bn(**kwargs):
  """VGG 11-layer model (configuration "A") with batch normalization
  """
  model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
  return model


def vgg13(**kwargs):
  """VGG 13-layer model (configuration "B")
  """
  model = VGG(make_layers(cfg['B']), **kwargs)
  return model


def vgg13_bn(**kwargs):
  """VGG 13-layer model (configuration "B") with batch normalization
  """
  model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
  return model


def vgg16(**kwargs):
  """VGG 16-layer model (configuration "D")
  """
  model = VGG(make_layers(cfg['D']), **kwargs)
  return model


def vgg16_bn(**kwargs):
  """VGG 16-layer model (configuration "D") with batch normalization
  """
  model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
  return model


def vgg19(**kwargs):
  """VGG 19-layer model (configuration "E")
  """
  model = VGG(make_layers(cfg['E']), **kwargs)
  return model


def vgg19_bn(**kwargs):
  """VGG 19-layer model (configuration 'E') with batch normalization
  """
  model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
  return model
