import argparse
import os
from functools import partial
from glob import glob
from zipfile import ZipFile

from tqdm.contrib.concurrent import process_map

# list of filenames that should exist in the zip file
ACCEPTABLE = [
    "ACC.csv",
    "BVP.csv",
    "EDA.csv",
    "HR.csv",
    "IBI.csv",
    "info.txt",
    "tags.csv",
    "TEMP.csv",
]


def rewrite(zip_file: str, output_dir: str):
    original = ZipFile(zip_file, "r")
    clean = ZipFile(os.path.join(output_dir, os.path.basename(zip_file)), "w")
    for item in original.infolist():
        # some corrupted zip has nested folders
        basename = os.path.basename(item.filename)
        if basename in ACCEPTABLE:
            buffer = original.read(item.filename)
            item.filename = basename
            clean.writestr(item, buffer)
    clean.close()
    original.close()


def main(args):
    # if args.output_dir is not defined, save zip files to args.input_dir/clean
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "clean")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    zip_files = sorted(glob(os.path.join(args.input_dir, "*.zip")))

    process_map(
        partial(rewrite, output_dir=args.output_dir),
        zip_files,
        max_workers=args.num_workers,
    )

    print(f"{len(zip_files)} clean zip files written to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=6)

    main(parser.parse_args())
