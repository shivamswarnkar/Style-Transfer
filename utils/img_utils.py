import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms
import numpy as np


def load_image(img_path, max_size=400, shape=None):
  if 'http' in img_path:
    response = requests.get(img_path)
    image = Image.open(BytesIO(response.content)).convert('RGB')
  else:
    image = Image.open(img_path).convert('RGB')
  
  # Resize if exceeds max_size
  size = min(max(image.size), max_size)
  
  if shape is not None:
    size = shape
    
  in_transform = transforms.Compose([
      transforms.Resize(size),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406),
                           (0.229,0.224, 0.225))])

  # discard alpha channel
  image = in_transform(image)[:3, :, :].unsqueeze(0)
  
  return image


def im_convert(tensor):
  '''
  helper function for un-normalizing an image
  and converting it from a Tensor image to 
  a Numpy image for display
  '''
  image = tensor.to('cpu').clone().detach()
  image = image.numpy().squeeze()
  image = image.transpose(1,2,0)
  image = image * np.array((0.229, 0.224, 0.225)) \
  + np.array((0.485, 0.456, 0.406))
  image = image.clip(0,1)
  
  return image