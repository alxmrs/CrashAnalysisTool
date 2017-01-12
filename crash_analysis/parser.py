import pandas as pd
import xml.etree.ElementTree as etree
from glob import glob
from os import walk
from os.path import isfile
from zipfile import ZipFile


def extract_zipfiles(zipfile_dir):
    """
    Takes in a directory with zipfiles. Extracts each file and outputs the result to a directory with the same name
    as the zipfile.
    :param zipfile_dir: Directory with zipfiles
    :return: None
    :size_effects: Creates directories with the contents of the original zipfiles
    """
    zipfiles = glob(zipfile_dir + '*.zip')

    for current_file in zipfiles:
        dest_dir = current_file.replace('.zip', '')

        with ZipFile(current_file, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)


def xmldocs_to_dataframe(xml_dir):
    """
    Converts xml documents to a Pandas Dataframe
    :param xml_dir: Takes in a directory that has xml files
    :return: A pd.DataFrame with the information from each xml file as a row
    """
    # map xml files in dir to list of strings
    # xml_files = glob(xml_dir + '**/crashrpt.xml')

    crash_dirs = list()

    for (dirpath, dirnames, filenames) in walk(xml_dir):
        crash_dirs.append(dirpath)

    crash_dirs.pop(0)  # remove first elem, will be parent dir

    data_tuples = []

    for dir in crash_dirs:
        tmp_list = list()

        crashrpt_path = dir + '\\crashrpt.xml'
        exception_path = dir + '\\ManagedException.txt'

        if isfile(crashrpt_path):
            tmp_list.append(crashrpt_path)

        if isfile(exception_path):
            tmp_list.append(exception_path)

        if len(tmp_list) != 0:
            data_tuples.append(tmp_list)

    if not data_tuples:
        raise AssertionError('xml_dir did not point to a directory with xml files! Please specify another path.')

    trees = [(__xml_to_tree(data_tuple[0]), __xml_to_tree(data_tuple[1])) if len(data_tuple) == 2
             else (__xml_to_tree(data_tuple[0]),)
             for data_tuple in data_tuples]

    # map xml_strs to single dataframe
    df = __trees_to_dataframe(trees)

    return df


def __trees_to_dataframe(roots):
    """
    Converts a list of ElementTree trees into a pd.DataFrame
    :param roots: list of ElementTree roots
    :return: a pd.DataFrame with each root representing a row
    """
    return pd.DataFrame(list(__parse_etrees(roots)))


def __parse_etrees(roots):
    """
    A generator function that parses the ElementTree or list of ElementTrees (for multiple files per row) and converts
    it to a dictionary.
    :param roots: roots to process
    :return: A data dictionary, one row at a time
    """
    # go through each root group
    for root in roots:

        data_dict = {}

        # go through each file in the root groups
        for src_file in root:

            # get interator to move through tree
            iterator = src_file.iter()

            # depth-first search through nodes, adding to dictionary
            for node in iterator:
                if 'name' in node.attrib:
                    data_dict[node.attrib['name']] = node.attrib['value'] if 'value' in node.attrib else node.attrib[
                        'description']
                else:
                    data_dict[node.tag] = node.text

            yield data_dict


def __xml_to_tree(xml_filename):
    """
    Opens an xml file, converts it to a python ElementTree object, returns the root of the tree
    :param xml_filename: xml file to parse
    :return: root of the xml tree
    """
    with open(xml_filename, 'r') as xml_file:
        # read the data and store it as a tree
        tree = etree.parse(xml_file)

        # get tree root
        return tree.getroot()



