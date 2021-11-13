import pandas as pd
import os


if __name__ == '__main__':
    filenames_vid = pd.read_excel('/Users/camilaroa/Downloads/videoList.xlsx', dtype={'PatientID': str})
    feature_vector = filenames_vid.copy()

    folder_vid = '/Users/camilaroa/Downloads/ParkinsonVideos'

    for _, filename in filenames_vid.iterrows():
        test_path = os.path.join(folder_vid, filename['PatientID'], filename['DateTimeStatus'], filename['Movement'] + '_' + filename['Hand'])
        csv_path = test_path + '.csv'
        picture_path = test_path + '.jpg'

        break