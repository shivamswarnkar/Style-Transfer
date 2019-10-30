from utils.args import get_args
from style_transfer.main import transfer

def style_transfer(	content_url, 
					style_url, 
					output_file='output/output.jpg', 
					ngpu=1, 
					content_weight=1, 
					style_weight=1e6, 
					save_every=400,
					make_new=False, 
					steps=2000):

	# get default args
	args = get_args(base_args=True)

	# update args 
	args.content_url = content_url
	args.style_url = style_url
	args.output_file = output_file
	args.ngpu = ngpu
	args.content_weight = content_weight
	args.style_weight = style_weight
	args.save_every = save_every
	args.make_new = make_new
	args.steps = steps

	# transfer style
	transfer(args)