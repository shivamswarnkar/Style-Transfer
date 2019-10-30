
import torch
import torch.optim as optim
from torchvision import transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.img_utils import load_image, im_convert
from utils.transfer_utils import get_features, gram_matrix

import argparse

def get_args():
	parser = argparse.ArgumentParser('Style Transfer')

	parser.add_argument('--content_url', type=str, 
		default=f'imgs/content{np.random.randint(1, 5)}.jpg', 
		help='Path/URL of Content Image')

	parser.add_argument('--style_url', type=str, 
		default=f'imgs/style{np.random.randint(1, 6)}.jpg',
		help='Path/URL of Style Image')

	parser.add_argument('--output_file', type=str, 
		default=f'output/output.jpg',
		help='Filename for the output file')

	parser.add_argument('--ngpu', type=int, 
		default=1, 
		help='Number of GPUs to use')

	parser.add_argument('--content_weight', type=float, 
		default=1, 
		help='alpha value, the weight of content image in style transfer')

	parser.add_argument('--style_weight', type=float, 
		default=1e6, 
		help='beta value, the weight of style image in style transfer')

	parser.add_argument('--save_every', type=int, 
		default=400, 
		help='How often should the target image be saved')

	parser.add_argument('--make_new', type=bool, 
		default=False, 
		help='If true, every time target image will be saved as a new file, without replacing the previously saved target image')

	parser.add_argument('--show_when_saved', type=bool, 
		default=False, 
		help='Should the target image be shown when show_when_saved. Recommended only if working in Jupyter Notebook')

	parser.add_argument('--steps', type=int, 
		default=2000, 
		help='Total Number of Steps for editing target image')

	args = parser.parse_args()	
	return args

def transfer(args):
	device = torch.device('cuda' if (torch.cuda.is_available() and args.ngpu > 0) else 'cpu')
	# load non-classifier layers (features)
	vgg = models.vgg19(pretrained=True).features

	# freeze all VGG params
	for param in vgg.parameters():
		param.requires_grad_(False)

	vgg.to(device)

	# load content & style image
	content = load_image(args.content_url).to(device)

	# resize style to match content, makes code easier
	style = load_image(args.style_url, shape=content.shape[-2:]).to(device)


	# get features
	content_features = get_features(content, vgg)
	style_features = get_features(style, vgg)

	# calculate grams
	style_grams = {layer:gram_matrix(style_features[layer]) 
                                 for layer in style_features}

    # initialize target from content
	target = content.clone().requires_grad_(True).to(device)


	# params
	style_weights = {'conv1_1':1.,
				'conv2_1':0.8,
                'conv3_1':0.5,
                'conv4_1':0.3,
                'conv5_1':0.1}

	

	# hyper params
	content_weight = args.content_weight #alpha
	style_weight = args.style_weight #beta
	optimizer = optim.Adam([target], lr=0.003)
	steps = args.steps

	# Starting Style Transfer
	print(f'starting style transfer of {args.style_url} into {args.content_url}')
	for ii in range(1, args.steps+1):
  		target_features = get_features(target, vgg)
  
  
  		# calculate content loss
  		content_loss = torch.mean((target_features['conv4_2']
  			- content_features['conv4_2'])**2)
  
  		# calculate style loss
  		style_loss = 0
  		for layer in style_weights:
  			target_feature = target_features[layer]
  			_, d, h, w = target_feature.shape

  			# target gram
  			target_gram = gram_matrix(target_feature)

  			# style gram
  			style_gram = style_grams[layer]

  			# style loss for curr layer
  			layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

  			style_loss += layer_style_loss / (d * h * w)

  		total_loss = ((content_weight * content_loss) + (style_weight * style_loss)) /(content_weight+style_weight)

  		optimizer.zero_grad()
  		total_loss.backward()
  		optimizer.step()

  		if ii % args.save_every == 0:
  			if args.make_new:
  				fname = f'output/checkpoint{ii//args.save_every}.jpg' 
  			else:
  				fname = args.output_file
  			
  			plt.imshow(im_convert(target))
  			plt.axis('off')
  			plt.savefig(fname,bbox_inches='tight')
  			print(f'Total loss:{total_loss.item():.4f}\tSaved as {fname}')
  			if args.show_when_saved:
  				plt.show()
	plt.imshow(im_convert(target))
	plt.axis('off')
	plt.savefig(args.output_file,bbox_inches='tight')
	plt.show()

if __name__ == '__main__':
	args = get_args()
	transfer(args)