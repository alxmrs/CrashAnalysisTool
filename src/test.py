from crash_analysis.parser import *

import os
def main():
    zip_loc = './tmp/'
    extract_zipfiles(zip_loc)
    # df = xmldocs_to_dataframe(zip_loc)
if __name__ == '__main__':
    main()