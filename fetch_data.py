import sys
sys.path.append('/opt/ml/processing/input') # To read helper files
from utils import get_data_from_dynamodb
import pandas as pd
import numpy as np
import argparse
import os



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='Date for which data needs to be fetched for training (yyyy-mm-dd) e.g. "2021-07-21"')
    parser.add_argument('--output-dir', type=str, help='Local directory (or in container) where fetched data is saved as parquet', default='/opt/ml/processing/output')
    
    # Parse arguments
    args, _ = parser.parse_known_args()
    
    date = args.date
    output_dir = args.output_dir
    
    # Get daata from dunamodb table, date entered should correspond to data that must be used for training
    table_name = 'hastie'
    data = get_data_from_dynamodb(date, table_name)
    
    df = pd.DataFrame(data)
    
    # Save raw data to be used for preprocessing in next step
    filepath = os.path.join(output_dir, 'raw_data.parquet')
    df.to_parquet(filepath)