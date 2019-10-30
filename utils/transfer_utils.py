import torch
def get_features(image, model, layers=None):
  if layers is None:
    layers = {'0':'conv1_1',
              '5':'conv2_1',
              '10':'conv3_1',
              '19':'conv4_1',
              '21':'conv4_2', # content repr
              '28':'conv5_1',}
    
  features = {}
  x = image
  for name, layer in model._modules.items():
    x = layer(x)
    if name in layers:
      features[layers[name]] = x
      
  return features


def gram_matrix(tensor):
  _, d, h, w = tensor.size()
  tensor = tensor.view(d, h*w)
  return torch.mm(tensor, tensor.t()) #gram