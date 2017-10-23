import pandas as pd
import xml.etree.ElementTree as etree
from glob import glob
import os
from os import walk
from os.path import isfile
import zipfile
from zipfile import ZipFile


def extract_zipfiles(zipfile_dir):
    """Extract all the zipfiles in a directory.
    
    Takes in a directory with zipfiles. Extracts each file and outputs the result to a directory with the same name
    as the zipfile.
    :param zipfile_dir: Directory with zipfiles
    :return: None
    :size_effects: Creates directories with the contents of the original zipfiles
    """
    zipfiles = glob(zipfile_dir + '**.zip', recursive=True)

    for current_file in zipfiles:
        dest_dir = current_file.replace('.zip', '')
        try:
            with ZipFile(current_file, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
        except zipfile.BadZipfile as e:
            print('BadZipFile: {0}'.format(current_file))


def xmldocs_to_dataframe(xml_dir):
    """Converts xml documents to a Pandas Dataframe.
    
    Makes use of all the private helper functions in the file.
    Includes specified list of XML files ('crashrpt.xml' and 'ManagedException.txt'). 

    :param xml_dir: Takes in a directory that has xml files
    :return: A pd.DataFrame with the information from each xml file as a row
    """
    files_to_include = ['crashrpt.xml', 'ManagedException.txt']

    data_tuples = []

    for (dirpath, dirnames, filenames) in walk(xml_dir):
        report_group = [os.path.join(dirpath, file) for file in filenames if file in files_to_include]
        data_tuples.append(report_group)

    if not data_tuples:
        raise AssertionError('xml_dir did not point to a directory with xml files! Please specify another path.')

    trees = [[__xml_to_tree(value) for value in data_tuple] for data_tuple in data_tuples]

    # map xml_strs to single dataframe
    df = __trees_to_dataframe(trees)

    return df


def __trees_to_dataframe(roots):
    """Converts a list of ElementTree trees into a pd.DataFrame
    
    Helper function to convert XML trees into a single dataframe. 
    
    :param roots: list of ElementTree roots
    :return: a pd.DataFrame with each root representing a row
    """
    return pd.DataFrame(list(__parse_etrees(roots)))


def __parse_etrees(roots):
    """Parse a list of XML trees into a dictionary. 
    
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
    """Convert xml file to a Tree representation (etree).
    
    Opens an xml file, converts it to a python ElementTree object, returns the root of the tree
    :param xml_filename: xml file to parse
    :return: root of the xml tree
    """
    with open(xml_filename, 'r') as xml_file:
        lines = xml_file.readlines()

    try:
        xml_unicode = ''.join(lines)
        xml_str = xml_unicode.encode('ascii', 'ignore')

        root = etree.fromstring(xml_str)
    except Exception as e:
        root = etree.fromstring('<empty></empty>')  # on error, return empty element tree
    finally:
        return root





