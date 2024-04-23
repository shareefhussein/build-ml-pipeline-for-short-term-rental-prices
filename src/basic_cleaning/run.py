#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import os
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logging.info(f"Download the artifact {args.input_artifact}")
    df = pd.read_csv(artifact_local_path)

    logger.info(f"Remove outliers (min-max): {args.min_price} - {args.max_price}")
    min_price = args.min_price
    max_price = args.max_price 
    idx = df['price'].between(min_price, max_price)

    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    tmp_dir  = os.path.join(args.tmp_dir,args.output_artifact)
    logger.info(f"save final cleaned artifact to: {tmp_dir}")
    df.to_csv(tmp_dir , index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file(tmp_dir)
    run.log_artifact(artifact)

    artifact.wait()
    logger.info("artifact done uploading!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="data cleaning step")

    parser.add_argument(
        "--tmp_dir",
        type=str,
        help="tmp directory",
        required=True
    )

    parser.add_argument(
        "--input_artifact", 
        type = str,
        help="directory to save artifacts to",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type = str,
        help = "output cleaned artifact (*.csv)",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type = str,
        help = "output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type = str,
        help = "output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type = float,
        help = "Minimum valid price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type = float,
        help = "Maximum valid price",
        required=True
    )

    args = parser.parse_args()

    go(args)
