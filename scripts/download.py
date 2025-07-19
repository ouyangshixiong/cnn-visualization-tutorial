"""Dataset download script for CNN visualization."""

import argparse
from src.datasets.downloader import DatasetDownloader


def main():
    """Download datasets for CNN visualization."""
    parser = argparse.ArgumentParser(description="Download datasets for CNN visualization")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=list(DatasetDownloader.SUPPORTED_DATASETS.keys()),
        help="Dataset to download"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to save datasets"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets"
    )
    
    args = parser.parse_args()
    
    if args.list:
        datasets = DatasetDownloader.list_datasets()
        print("Available datasets:")
        for name, desc in datasets.items():
            print(f"  {name}: {desc}")
    else:
        print(f"Downloading {args.dataset} dataset...")
        DatasetDownloader.download(args.dataset, args.data_dir)
        print(f"{args.dataset} dataset downloaded successfully!")


if __name__ == "__main__":
    main()