import argparse

from training import train, test


def main():
    ap = argparse.ArgumentParser("Progressive Transformers")
    ap.add_argument("mode", choices=["train", "test"], help="train a model or test")
    ap.add_argument("config_path", type=str, help="path to YAML config file")
    ap.add_argument("--ckpt", type=str, help="path to model checkpoint")
    args = ap.parse_args()
    print("Received arguments:", args)  # Debug print
    if args.mode == "train":
        train(cfg_file=args.config_path, ckpt=args.ckpt)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
