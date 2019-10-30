import argparse
import numpy as np
def get_args(base_args=False):
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

	parser.add_argument('--steps', type=int, 
		default=2000, 
		help='Total Number of Steps for editing target image')

	if base_args:
		args = parser.parse_args([])	
	else:
		args = parser.parse_args()
	return args