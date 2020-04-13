import os
import urllib
import zipfile

def download_and_unzip_datasets(zip_url, zip_path):
    '''
    download and unzip a zipped file from Figshare
    
    returns a directory
    '''
    
    urllib.request.urlretrieve(zip_url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()