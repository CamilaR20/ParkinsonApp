import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from zipfile import ZipFile

if __name__ == '__main__':
    # Connect to Firebase
    cred = credentials.Certificate('/Users/camilaroa/PycharmProjects/parkinsonApp/serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {'storageBucket': 'parkinsondata.appspot.com'})

    bucket = storage.bucket()
    blob = bucket.blob("1111/14082021_1702/test.zip")

    # Download file from Firebase
    # blob.download_to_filename("/Users/camilaroa/Downloads/intento1.zip")
    print("Downloaded storage object to local file.")

    # List files in Firebase
    blobs = bucket.list_blobs()
    for blob in blobs:
        print(blob.name)

    # Zip test
    # test_file_name = "/Users/santiagorojasjaramillo/Downloads/intento1.zip"
    # with ZipFile(test_file_name, 'r') as zip:
    #     zip.printdir()
    #     zip.extractall('/Users/santiagorojasjaramillo/Downloads/temp')