from crash_analysis.parser import *


def main():
    zip_loc = 'C:\\CrashReports\\'

    df = xmldocs_to_dataframe(zip_loc)
if __name__ == '__main__':
    main()