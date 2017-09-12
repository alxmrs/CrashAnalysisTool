from crash_analysis.parser import *
from crash_analysis import downloader
from crash_analysis.downloader import BASE_DIR

import datetime
import os
def main():

    # now = datetime.datetime.now()
    # start = datetime.datetime(2017,3,8)
    # downloader.download_time_range(start, now)

    zipfile_location = os.path.join(BASE_DIR, 'tmp' + os.sep)
    # zip_loc = 'tmp/'
    #
    extract_zipfiles(zipfile_location)
    df = xmldocs_to_dataframe(zipfile_location)

    print(df.head())

    print('without duplicates')
    df.drop_duplicates(inplace=True)
    print(df.head())
    print(len(df))

    print(df.columns)
if __name__ == '__main__':
    main()