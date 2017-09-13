from crash_analysis import QB
from crash_analysis import Client
from itertools import repeat
from multiprocessing import Pool

import multiprocessing
import datetime
import urllib.request as req
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

default_dest = os.path.join(BASE_DIR, 'tmp' + os.sep)

DEBUG = True

optimal_thread_count = multiprocessing.cpu_count() + 1

def test_query():

    now = datetime.datetime.now()
    start = now + datetime.timedelta(days=-1)
    end = now

    download_time_range(start, end)

    if(DEBUG):
       print('\n\n\nFinished.')

def download_time_range(start, end, dest=None):
    """

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


def download_file(url, ticket, dest=None):
    if not dest:
        dest = default_dest

    filename = url.split('/')[-1]

    req_url = url + '?ticket=' + ticket

    req.urlretrieve(req_url, dest + filename)
    if(DEBUG):
        print('Started download of ' + filename)



if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    test_query()