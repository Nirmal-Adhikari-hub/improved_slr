from cslr.utils.parse_args import parse_args
from box.box import Box


if __name__ == "__main__":
    args = Box(parse_args())
    cfg = args.config
    print(f"Configuration file: {cfg.dataset_info.evaluate_dir}")