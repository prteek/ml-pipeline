import sys

sys.path.append("/opt/ml/processing/input")  # To read helper files
from utils import get_data_from_dynamodb, make_simple_pipe, unpack_nested_list
import pandas as pd
import argparse
import os
from logger import logger
from datetime import datetime, timedelta

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--end-date",
        type=str,
        help='Last Date from which look back will start to fetch data for training (yyyy-mm-dd) e.g. "2021-07-21"',
    )
    parser.add_argument(
        "--number-of-days",
        type=str,
        help='Number of days looking back from the date for which data needs to be fetched e.g. "7"',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Local directory (or in container) where fetched data is saved as parquet",
        default="/opt/ml/processing/output",
    )

    # Parse arguments
    args, _ = parser.parse_known_args()

    date = datetime.strptime(args.end_date, "%Y-%m-%d")
    number_of_days = eval(args.number_of_days)
    output_dir = args.output_dir

    # Get daata from dunamodb table, dates should correspond to data that must be used for training
    table_name = "hastie"

    date_list = list(
        map(
            lambda i: (date - timedelta(days=i)).strftime("%Y-%m-%d"),
            range(number_of_days),
        )
    )  # Look back from end date

    get_data_from_date_list = lambda date_list: list(
        map(lambda d: get_data_from_dynamodb(d, table_name), date_list)
    )

    get_data_recursively = make_simple_pipe(
        [get_data_from_date_list, unpack_nested_list]
    )

    logger.info("Getting data from dynamodb")
    data = get_data_recursively(date_list)
    df = pd.DataFrame(data)

    # Save raw data to be used for preprocessing in next step
    logger.info("Saving raw data as parquet")
    filepath = os.path.join(output_dir, "raw_data.parquet")
    df.to_parquet(filepath)
