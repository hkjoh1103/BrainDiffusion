
# %%
# import library
import argparse

from diffusion.train import *
from diffusion.sample import *

# %%
## Parser genration
parser = argparse.ArgumentParser(description="BrainDiffusion",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=["train", "test", "sample"], type=str, dest="mode")
# parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")
parser.add_argument("--name", required=True, type=str, dest="name")

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=16, type=int, dest="batch_size")
parser.add_argument("--num_iteration", default=100, type=int, dest="num_iteration")
parser.add_argument("--dropout", default=0.1, type=float, dest="dropout")

parser.add_argument("--time_step", default=300, type=int, dest="time_step")
parser.add_argument("--schedule", default='cosine', type=str, dest="schedule")
parser.add_argument("--schedule_low", default=1e-4, type=float, dest="schedule_low")
parser.add_argument("--schedule_high", default=0.02, type=float, dest="schedule_high")
parser.add_argument("--image_size", default=32, type=int, dest="image_size")
parser.add_argument("--base_channels", default=64, type=int, dest="base_channels")
parser.add_argument("--channel_mults", default=(1,2,4,8), type=tuple, dest="channel_mults")
parser.add_argument("--num_res_blocks", default=2, type=int, dest="num_res_blocks")
parser.add_argument("--time_emb_dim", default=128*2, type=int, dest="time_emb_dim")

parser.add_argument("--ema_decay", default=0.9999, type=float, dest="ema_decay")
parser.add_argument("--ema_update_rate", default=1, type=int, dest="ema_update_rate")

parser.add_argument("--data_dir", default="/mnt/ssd/bhkim/ukb/nifti/flair/flair", type=str, dest="data_dir")
# parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
# parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
# parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--log_rate", default=10, type=int, dest="log_rate")
parser.add_argument("--save_rate", default=1000, type=int, dest="save_rate")

args = parser.parse_args()


if __name__ == "__main__":
    if args.mode == "train":
        try:
            train(args)
        except KeyboardInterrupt:
            print('Erroor: Keyboard Interrupt')
    elif args.mode == "test" or "sample":
        try:
            sample(args)
        except KeyboardInterrupt:
            print('Erroor: Keyboard Interrupt')
# %%

