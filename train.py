"""Explicit entrypoint for the supplied modified Ultralytics source."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "val", "predict"])
    parser.add_argument("--model", required=True, help="Model YAML for training or original checkpoint")
    parser.add_argument("--data", help="Your actual five-class dataset YAML")
    parser.add_argument("--source", help="Input image directory for predict")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if not Path(args.model).is_file(): parser.error("--model must point to an existing YAML/checkpoint")
    if args.mode in {"train", "val"} and (not args.data or not Path(args.data).is_file()):
        parser.error("train/val requires an existing --data YAML")
    if args.mode == "predict" and (not args.source or not Path(args.source).exists()):
        parser.error("predict requires an existing --source")
    from ultralytics import YOLO
    model = YOLO(args.model)
    common = dict(device=args.device, imgsz=args.imgsz)
    if args.mode == "train": model.train(data=args.data, epochs=args.epochs, workers=args.workers, **common)
    elif args.mode == "val": model.val(data=args.data, workers=args.workers, **common)
    else: model.predict(source=args.source, save=True, **common)

if __name__ == "__main__": main()
