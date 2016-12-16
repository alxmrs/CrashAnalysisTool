from __future__ import print_function
import pandas as pd
import CrashAnalysis


def main():
   pd.set_option('display.height', 5000)
   pd.set_option('display.max_rows', 5000)

   crash_file = 'C:\\Dev\\CrashAnalysis\\src\\data\\Crashes2.csv'
   pro_v_basic = False
   total = False
   lda = False
   force_recompute = False
   parse_xml = True
   extract_zipfiles = False
   tool = CrashAnalysis.TextAnalysis(crash_file)


   if extract_zipfiles:
      xml_parser = CrashAnalysis.CrashReportParser()
      xml_parser.extract_zipfiles('C:\\CrashReports\\')

   if parse_xml:
      xml_parser = CrashAnalysis.CrashReportParser()
      xml_df = xml_parser.xmldocs_to_dataframe('C:\\CrashReports\\')

      # print(xml_df['InstallType'][:10])
      print(xml_df.columns)
      '''
          Index([u'ACCDT_Field', u'Active_ClientFileName', u'Active_Field',
          u'Active_Form', u'Active_FormsetID', u'Active_FormsetVersion',
          u'AppName', u'AppVersion', u'BasWin16.INI', u'Batch_ClientFileName',
          u'CrashGUID', u'Current_Calcsection', u'CustNum', u'DataFileCount',
          u'ExceptionAddress', u'ExceptionCode', u'ExceptionModule',
          u'ExceptionModuleBase', u'ExceptionModuleVersion', u'ExceptionType',
          u'FormsPrinter', u'GUIResourceCount', u'GeoLocation', u'ImageName',
          u'InstallType', u'Last_Calcsection', u'ManagedException.txt',
          u'MemoryUsageKbytes', u'OSIs64Bit', u'OpenHandleCount',
          u'OperatingSystem', u'ProWin16.INI', u'ProblemDescription',
          u'SystemTimeUTC', u'WorkStationName', u'WorkStationType',
          u'crashdump.dmp', u'crashrpt.xml'],
          dtype='object')
      '''

      # print(xml_df.fillna(' ').groupby(['InstallType', 'WorkStationType', 'OperatingSystem'])['SystemTimeUTC'].count())
      # print(xml_df.fillna(' ').groupby('InstallType')['ExceptionAddress'].apply(lambda x: '9be7' in x))




   if pro_v_basic:
      pro_name = 'ProSeries - 2016'
      basic_name = 'ProSeries Basic Edition - 2016'
      print(pro_name)
      tool.frequency('2016040014', product_id=pro_name, top=50)
      if lda:
         model = tool.lda('2016040014', product_id=pro_name, recompute=force_recompute, num_topics=10)
         tool.print_topics(model, num_words=10)

      print()
      print(basic_name)
      tool.frequency('2016040014', product_id=basic_name, top=50)

      if lda:
         model = tool.lda('2016040014', product_id=basic_name, recompute=force_recompute, num_topics=10)
         tool.print_topics(model, num_words=10)


   if total:

      vocab, sortedFreq = tool.frequency('2016040014', print_output=False, top=50)

      print(tool.get_columns())

      if lda:
         model = tool.lda('2016040014', recompute=force_recompute, num_topics=10)
         tool.print_topics(model, num_words=10)


      err_codes = tool.group_by_vocab(vocab, sortedFreq, tool.df, field='Payload')

      print('Error Codes by Keyterm')
      for word, count in sortedFreq:
         print('keyterm: ' + word)
         print(err_codes[word][err_codes[word] > 0])

         if count < 10:
            break

   # mx, terms = tool.vectorize_corpus()
   #
   #
   #
   # compute = True
   # n_custers = 7
   # if compute:
   #     km = tool.kmeans(mx, n_custers)
   # else:
   #     km = joblib.load('doc_cluster_k{0}.pkl'.format(n_custers))
   #
   # clusters = km.labels_.tolist()
   # cluster_lists = [[x] for x in clusters]
   #
   # print(tool.frequency_count(cluster_lists))
   #
   # new_df = tool.label_dataframe_with_clusters(clusters)
   #
   # print(new_df['Cluster'].value_counts())
   #
   # # tool.label_frame_with_clusters(clusters)
   #
   # top_terms_per_cluster(new_df, km, n_custers, vocab_frame, terms)


def top_terms_per_cluster(frame, km, num_clusters, vocab_frame, terms):
   print("Top terms per cluster:")
   print()

   # sort cluster centers by proximity to centroid
   order_centroids = km.cluster_centers_.argsort()[:, ::-1]

   for i in range(num_clusters):
      print("Cluster %d words:" % i, end='')

      for ind in order_centroids[i, :10]:  # replace 6 with n words per cluster
         print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
      print()
      print()

      # print("Cluster %d titles:" % i, end='')
      # for err_code in frame.ix[i]['Error_Code'].values.tolist():
      #     print(' %s,' % err_code, end='')
      # print()
      # print()

   print()
   print()


if __name__ == '__main__':
   main()
