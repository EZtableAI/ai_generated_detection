# This file create some tools to preprocess text file
import os
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from sklearn.model_selection import train_test_split

def get_data_from_kaggle():
    # Copy kaggle.json to the appropriate directory
    subprocess.run(['cp', 'kaggle.json', os.path.expanduser('~/.kaggle/')])

    # Download the dataset from Kaggle
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'thedrcat/daigt-proper-train-dataset'])

    # Create the dataset directory
    os.makedirs('dataset', exist_ok=True)

    # Unzip the dataset into the dataset directory
    subprocess.run(['unzip', 'daigt-proper-train-dataset.zip', '-d', 'dataset'])

    # Rename the CSV file
    os.rename('dataset/train_drcat_01.csv', 'dataset/data_01.csv')
    os.rename('dataset/train_drcat_02.csv', 'dataset/data_02.csv')
    os.rename('dataset/train_drcat_03.csv', 'dataset/data_03.csv')
    os.rename('dataset/train_drcat_04.csv', 'dataset/data_04.csv')

    # Remove the zip file and other unwanted CSV files to optimize memory
    os.remove('daigt-proper-train-dataset.zip')

def clean_text(text):
    # define all signal
    escape_characters = {
        '\\': '',
        '\'': '',
        '\"': '',
        '\n': ' ',
        '\t': ' ',
        '\r': ' ',
        '\b': '',
        '\f': ' ',
        '\v': ' ',
        '\a': '',
        '\0': '',
    }
    
    # replace all
    for char, replacement in escape_characters.items():
        text = text.replace(char, replacement)
    
    return text

def combin_file(folder_path, max=100000):
    # load all csv file
    dataframes = []
    filename_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # read csv file
            df = pd.read_csv(file_path)
            dataframes.append(df)
            filename_list.append(filename)
    # concat all df
    combined_df = pd.concat(dataframes, ignore_index=True)
    final_df = combined_df.groupby('label').apply(lambda x: x.sample(n=max, random_state=1)).reset_index(drop=True)
    final_df['cleaned_text'] = final_df['text'].apply(clean_text)
    X = final_df['cleaned_text']
    y = final_df['label']
    y = y.astype('int')
    y = y.astype('str')
    
    return X, y, filename_list

def save_to_txt(X, y, prefix):
    # train test valid splite
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
    all_x = [X_train, X_test, X_valid]
    all_y = [y_train, y_test, y_valid]
    mode = ['train', 'test', 'valid']

    # save all data to txt file
    for x, y, mode in zip(all_x, all_y, mode, ):
        txt_file = f'{prefix}_{mode}_content.txt'
        label_file = f'{prefix}_{mode}_label.txt'

        all_text = ''
        for i in x:
            all_text += f'{i}\n'
        with open(txt_file, 'w') as txtfile:
            txtfile.write(all_text)
            print(f"Successfully save {txt_file}")
        
        all_lable = ''
        for i in y:
            all_lable += f'{i}\n'
        with open(label_file, 'w') as labelfile:
            labelfile.write(all_lable)
            print(f"Successfully save {label_file}")

def preprocess_file():

    # config
    folder_path = 'dataset'
    file_name = "data"
    prefix = Path(folder_path).absolute() / file_name

    # get content and label
    X, y, filename_list = combin_file(folder_path, 100)

    # train test valid splite & save all file to dataset
    save_to_txt(X, y, prefix)
    
    # remove csv file
    for filename in filename_list:
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
                