import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="replace data_root")

    parser.add_argument(
        "--json_file",
        help="data_json",
        required=False,
        type=str,
        default="/home/lijun/papers/NCL/dataset_json/Places_LT_train.json"
    )

    parser.add_argument(
        "--find_root",
        "-f",
        help="data root needs to be replaces",
        required=False,
        type=str,
        default="/media/test"
    )
    parser.add_argument(
        "--replaces_to",
        "-r",
        help="target root",
        type=str,
        required=False,
        default="/media/ssd2/lijun/data/places365_standard"
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    with open(args.json_file, "r") as f:
        info = json.load(f)
    for annotations in info['annotations']:
        annotations['fpath'] = annotations['fpath'].replace(args.find_root, args.replaces_to)
    with open(args.json_file, 'w') as f:
        json.dump(info, f)
    print("data root replace done")
    a = 0