from utils.args import get_args
from style_transfer.main import transfer

if __name__ == '__main__':
	# read arguments from terminal
	args = get_args()

	# transfer style
	transfer(args)