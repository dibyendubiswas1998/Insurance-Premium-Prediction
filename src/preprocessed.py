from src.utils.common_utils import log, save_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def target_encoding(data, xcol, dct_map, log_file):
    """
        It helps to apply target_encoding.\n
        :param data: data.csv
        :param col: column
        :param dct_map: mapping dictionary
        :param log_file: log_file.txt
        :return: data
    """
    try:
        data = data
        file = log_file
        log(file_object=file, log_message=f"perform target encoding encoding on xcol: {xcol}")  # logs the details

        data[xcol] = data[xcol].map(dct_map)
        return data # return data after applying target encoding.

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the details
        raise e


def mean_encoding(data, xcol, ycol, log_file):
    """
        It helps to apply mean_encoding.\n
        :param data: data
        :param xcol: xcol
        :param ycol: ycol
        :param log_file: log_file.txt
        :return: data, dct
    """
    try:
        data = data
        file = log_file
        log(file_object=file, log_message=f"perform Mean encoding on xcol: {xcol}, ycol: {ycol}")  # logs the details

        dct = data.groupby([xcol])[ycol].mean().sort_values(ascending=False).to_dict() # get dictionary
        data = target_encoding(data=data, xcol=xcol, dct_map=dct, log_file=log_file)
        return data, dct # return data & mapping dicts after applying mean encoding

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the details
        raise e


# def find_outliers(data, col, img_path, log_file):
#     """
#         It helps to find the outliers using IQR & also try save the box plot.\n
#         :param data: data
#         :param col: col
#         :param img_path: img_path
#         :param log_file: log_file
#         :return: None
#     """
#     try:
#         data = data
#         col = col
#         img_path = img_path
#         file = log_file
#         log(file_object=file, log_message=f"finding outliers using box plot")  # logs the details
#
#         sns.boxplot(data=data, x=col, color='darkblue') # plot the box plot individually.
#         plt.savefig(img_path) # save the file in png format.
#         log(file_object=file, log_message=f"save the box ploy in path: {img_path}")  # logs the details
#
#     except Exception as e:
#         print(e)
#         file = log_file
#         log(file_object=file, log_message=f"Error will be: {e}")  # logs the details
#         raise e


def handle_outliers(data, col, log_file):
    """
        It helps to remove the outliers using IQR method.\n
        :param data: data
        :param col: col
        :param log_file: log_file
        :return: data
    """
    try:
        data = data
        col = col
        file = log_file
        log(file_object=file, log_message=f"Handle the Outliers using IQR method, col: {col}")  # logs the details

        q1 = data[col].quantile(0.25) # 25-percentile
        q3 = data[col].quantile(0.75) # 75-percentile
        IQR = q3 - q1 # inter quantile range
        lower = q1 - 1.5 * IQR # lower limit
        upper = q3 + 1.5 * IQR # upper limit

        data.loc[data[col] >= upper, col] = upper
        data.loc[data[col] <= lower, col] = lower
        return data # return data after removing outliers

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the details
        raise e
