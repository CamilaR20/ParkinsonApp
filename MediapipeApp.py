import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from zipfile import ZipFile

cred = credentials.Certificate('/Users/santiagorojasjaramillo/Desktop/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'parkinsondata.appspot.com'})

bucket = storage.bucket()
blob = bucket.blob("1111/14082021_1702/test.zip")
blob.download_to_filename("/Users/santiagorojasjaramillo/Downloads/intento1.zip")

blobs = bucket.list_blobs()
for blob in blobs:
    print(blob.name)

test_file_name = "/Users/santiagorojasjaramillo/Downloads/intento1.zip"

with ZipFile(test_file_name, 'r') as zip:
    zip.printdir()
    zip.extractall('/Users/santiagorojasjaramillo/Downloads/temp')

