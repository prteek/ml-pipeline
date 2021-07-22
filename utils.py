from decimal import Decimal
from boto3.dynamodb.conditions import Key
import boto3
from functools import reduce
import io
import numpy as np
import json
import math


def write_dict_to_dynamodb(results_dict: dict, table_name: str) -> dict:
    """Convert a dictionary to json and write to dynamodb table with appropriate data types
    Examples:
    --------
    >>> table = 'results_table'
    >>> results = {'a':1.0, b: 'b_string'}
    >>> write_dict_to_dynamodb(results, results_table)
    """

    table = boto3.resource(
        "dynamodb"
    ).Table(table_name)

    data = json.loads(json.dumps(dict_to_dynamodb(results_dict)), parse_float=Decimal)
    response = table.put_item(Item=data)

    return response


def dict_to_dynamodb(item: dict) -> dict:
    """Convert datatypes of dictionary values to those allowed by dynamodb"""

    if isinstance(item, dict):
        return {k: dict_to_dynamodb(v) for k, v in item.items()}
    elif isinstance(item, list) or isinstance(item, np.ndarray):
        return [dict_to_dynamodb(l) for l in item]
    elif isinstance(item, float) and math.isnan(item):
        return str(item)
    elif (
        isinstance(item, np.float32)
        or isinstance(item, np.float64)
        or isinstance(item, np.int64)
        or isinstance(item, np.int8)
        or isinstance(item, np.uint8)
    ):
        return float(item)
    elif isinstance(item, np.bool_):
        return bool(item)
    else:
        return item
    
    

def dynamodb_to_dict(item):
    """Take json response from a dynamodb query and return a dictionary of appropriate data types"""

    if isinstance(item, dict):
        return {k: dynamodb_to_dict(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [dynamodb_to_dict(l) for l in item]
    elif isinstance(item, Decimal):
        if float(item) % 1 > 0:
            return float(item)
        else:
            return int(item)
    else:
        return item



def get_data_from_dynamodb(
    key_value,
    table_name,
    primary_key_name="date",
):
    """Convert a json response from dynamodb query to a dictionary"""

    table = boto3.resource(
        "dynamodb"
    ).Table(table_name)

    response = table.query(KeyConditionExpression=Key(primary_key_name).eq(key_value))
    results_dict = dynamodb_to_dict(response["Items"])

    return results_dict