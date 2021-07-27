import sys
sys.path.append('/opt/ml/processing/input') # To read helper files
import boto3
import argparse
from logger import logger


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, help="message to display and/or send as notification mail", default="Error/alert in step")
    
    args, _ = parser.parse_known_args()
    message = args.message
    
    logger.info("message")
