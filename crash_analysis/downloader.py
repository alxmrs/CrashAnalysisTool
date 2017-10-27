from crash_analysis import QB
from crash_analysis import Client
from crash_analysis.types import PathStr
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, cast
import multiprocessing
import datetime
import urllib.request as req
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

default_dest = cast(PathStr, os.path.join(BASE_DIR, 'tmp' + os.sep))

DEBUG = True

optimal_thread_count = multiprocessing.cpu_count() + 1


def download_time_range(start: datetime.datetime, end: datetime.datetime, dest: Optional[PathStr] = None) -> None:
    """Downloads all crashes from quickbase from the start time, end time, to the specified directory destination.

    Example:
    >>> import datetime
    >>> now = datetime.datetime.now()
    >>> start = now + datetime.timedelta(hours=-1)
    >>> download_time_range(start, end)
    """
    client = Client(**QB)
    records = client.do_query("{'1'.GTE.'%s'}AND{'1'.LT.'%s'}" % (start.isoformat(), end.isoformat()),
                              structured=True, include_rids=True, columns=[14], path_or_tag='./table/records/record/f/url'
                             )

    if (DEBUG):
        print('About to download {0} records.'.format(len(records)))

    with Pool(processes=optimal_thread_count) as pool:
        pool.starmap(download_file, zip(map(lambda r: r['url'], records), repeat(client.ticket), repeat(dest)))


def download_file(url: str, ticket: str, dest: Optional[PathStr] = None) -> None:
    """Download a single file from quickbase 
    
    Will skip downloading the file if it already exists on the file system. 
    If dest is not specified, will use a default destination. 
    Set DEBUG to true to see download status in stdout. 
    """
    if not dest:
        dest =  default_dest

    if not os.path.isdir(dest):
        os.mkdir(dest)

    filename = url.split('/')[-1]

    localfile = Path(dest + filename)

    if localfile.exists():
        if(DEBUG):
            print('{0} is already downloaded.'.format(filename))
        return

    req_url = url + '?ticket=' + ticket

    req.urlretrieve(req_url, dest + filename)
    if(DEBUG):
        print('Started download of ' + filename)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
