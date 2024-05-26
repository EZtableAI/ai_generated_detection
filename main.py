from data_preprocessing import get_data_from_kaggle, preprocess_file

# some config
file_name = 'data'
data_dir = '/dataset'


# prepare dataset
get_data_from_kaggle()
preprocess_file()
