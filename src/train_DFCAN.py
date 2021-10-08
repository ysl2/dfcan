import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=1)
parser.add_argument("--gpu_memory_fraction", type=float, default=0.3)
parser.add_argument("--mixed_precision_training", type=int, default=1)
parser.add_argument("--data_dir", type=str, default="../dataset/train/F-actin")
parser.add_argument("--save_weights_dir", type=str, default="../trained_models")
parser.add_argument("--model_name", type=str, default="DFCAN")
parser.add_argument("--patch_height", type=int, default=128)
parser.add_argument("--patch_width", type=int, default=128)
parser.add_argument("--input_channels", type=int, default=9)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--norm_flag", type=int, default=1)
parser.add_argument("--iterations", type=int, default=1000000)
parser.add_argument("--sample_interval", type=int, default=1000)
parser.add_argument("--validate_interval", type=int, default=2000)
parser.add_argument("--validate_num", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--start_lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--load_weights", type=int, default=0)
parser.add_argument("--optimizer_name", type=str, default="adam")

args = parser.parse_args()
gpu_id = str(args.gpu_id)
gpu_memory_fraction = args.gpu_memory_fraction
mixed_precision_training = str(args.mixed_precision_training)
data_dir = args.data_dir
save_weights_dir = args.save_weights_dir
validate_interval = args.validate_interval
batch_size = args.batch_size
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
patch_height = args.patch_height
patch_width = args.patch_width
input_channels = args.input_channels
scale_factor = args.scale_factor
norm_flag = args.norm_flag
validate_num = args.validate_num
iterations = args.iterations
load_weights = args.load_weights
optimizer_name = args.optimizer_name
model_name = args.model_name
sample_interval = args.sample_interval

