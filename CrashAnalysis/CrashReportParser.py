import datetime
import pandas as pd
import xml.etree.ElementTree as etree
from glob import glob

class CrashReportParser:
   def __init__(self):
      pass

   def xmldocs_to_dataframe(self, xml_dir):
      """
      Converts xml documents to a Pandas Dataframe
      :param xml_dir: Takes in a directory that has xml files
      :return: A pd.DataFrame with the information from each xml file as a row
      """
      # map xml files in dir to list of strings
      xml_files = glob(xml_dir + '**/*.xml')

      trees = [self._xml_to_tree(xml_file) for xml_file in xml_files]

      # map xml_strs to single dataframe
      df = self._trees_to_dataframe(trees)

      return df

   def _xml_to_tree(self, xml_filename):
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

   def _trees_to_dataframe(self, roots):
      """
      Converts a list of ElementTree trees into a pd.DataFrame
      :param roots: list of ElementTree roots
      :return: a pd.DataFrame with each root representing a row
      """
      return pd.DataFrame(list(self._parse_etrees(roots)))


   def _parse_etrees(self, roots):
      """
      A generator function that parses the ElementTree and converts it to a dictionary.
      :param roots: roots to process
      :return: A data dictionary, one row at a time
      """
      for root in roots:

         data_dict = {}

         for node in root:

            if '\n' not in node.text:
               data_dict[node.tag] = node.text

            else:
               for var in node:
                  data_dict[var.attrib['name']] = var.attrib['value'] if 'value' in var.attrib else var.attrib['description']

         yield data_dict
