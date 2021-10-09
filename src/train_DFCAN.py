import argparse

parser = argparse.ArgumentParser()
# --gpu_id: the gpu device you want to use in current task
parser.add_argument("--gpu_id", type=int, default=1)
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
parser.add_argument("--gpu_memory_fraction", type=float, default=0.3)
# --mixed_precision_training: whether use mixed precision training or not
parser.add_argument("--mixed_precision_training", type=int, default=1)
# --data_dir: the root path of training data folder
parser.add_argument("--data_dir", type=str, default="../dataset/train/F-actin")
# --save_weights_dir: root directory where models weights will be saved in
parser.add_argument("--save_weights_dir", type=str, default="../trained_models")
# --model_name: 'DFCAN' or 'DFGAN'
parser.add_argument("--model_name", type=str, default="DFCAN")
# --patch_height: the height of input image patches
parser.add_argument("--patch_height", type=int, default=128)
# --patch_width: the width of input image patches
parser.add_argument("--patch_width", type=int, default=128)
# --input_channels: 1 for WF input and 9 for SIM reconstruction
parser.add_argument("--input_channels", type=int, default=9)
# --scale_factor: 2 for linear SIM and 3 for nonlinear SIM
parser.add_argument("--scale_factor", type=int, default=2)

parser.add_argument("--norm_flag", type=int, default=1)
# --iterations: total training iterations
parser.add_argument("--iterations", type=int, default=1000000)

parser.add_argument("--sample_interval", type=int, default=1000)

parser.add_argument("--validate_interval", type=int, default=2000)

parser.add_argument("--validate_num", type=int, default=500)
# --batch_size: batch size for training
parser.add_argument("--batch_size", type=int, default=4)
# --start_lr: initial learning rate of training, typically set as 10-4
parser.add_argument("--start_lr", type=float, default=1e-4)
# --lr_decay_factor: learning rate decay factor, typically set as 0.5
parser.add_argument("--lr_decay_factor", type=float, default=0.5)

parser.add_argument("--load_weights", type=int, default=0)

parser.add_argument("--optimizer_name", type=str, default="adam")

args = parser.parse_args()
gpu_id = str(args.gpu_id)
gpu_memory_fraction = args.gpu_memory_fraction
mixed_precision_training = str(args.mixed_precision_training)
data_dir = args.data_dir
save_weights_dir = args.save_weights_dir
model_name = args.model_name
patch_height = args.patch_height
patch_width = args.patch_width
input_channels = args.input_channels
scale_factor = args.scale_factor
norm_flag = args.norm_flag
iterations = args.iterations
sample_interval = args.sample_interval
validate_interval = args.validate_interval
validate_num = args.validate_num
batch_size = args.batch_size
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
load_weights = args.load_weights
optimizer_name = args.optimizer_name

data_name = data_dir.split(os.sep)[-1]
if input_channels == 1:
    save_weights_name = model_name + '-SISR_' + data_name
    cur_data_loader = data_loader
    train_images_path = data_dir + '/training_wf/'
    validate_images_path = data_dir + '/validate_wf/'
else:
    save_weights_name = model_name + '-SIM_' + data_name
    cur_data_loader = data_loader_multi_channel
    train_images_path = data_dir + '/training/'
    validate_images_path = data_dir + '/validate/'
save_weights_path = save_weights_dir + '/' + save_weights_name + '/'
train_gt_path = data_dir + '/training_gt/'
validate_gt_path = data_dir + '/validate_gt/'
sample_path = save_weights_path + 'sampled_img/'

if not os.path.exists(save_weights_path):
    os.mkdir(save_weights_path)
if not os.path.exists(sample_path):
    os.mkdir(sample_path)
