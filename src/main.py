from __future__ import print_function
import pandas as pd
import CrashAnalysis


def main():
   pd.set_option('display.height', 5000)
   pd.set_option('display.max_rows', 5000)
   pd.options.display.max_colwidth = 250

   _crash_file = 'C:\\Dev\\CrashAnalysis\\src\\data\\Crashes3.csv'
   _pro_v_basic = False
   _total = False
   _lda = False
   _force_recompute = False
   _extract_zipfiles = False
   _parse_xml = True
   _merge_tables = True  # dependent on parse_xml
   _merge_tables_text_analysis = True


   tool = CrashAnalysis.TextAnalysis(_crash_file)


   if _extract_zipfiles:
      xml_parser = CrashAnalysis.CrashReportParser()
      xml_parser.extract_zipfiles('C:\\CrashReports\\')

   if _parse_xml:
      xml_parser = CrashAnalysis.CrashReportParser()
      xml_df = xml_parser.xmldocs_to_dataframe('C:\\CrashReports\\')




   if _merge_tables:

      xml_df['CrashGUID'] = xml_df['CrashGUID'].apply(lambda x: str(x) + '.zip')
      oldcols = list(tool.df.columns)
      oldcols[5] = 'CrashGUID'
      tool.df.columns = oldcols

      total_df = pd.merge(xml_df, tool.df, on='CrashGUID')
      total_df = total_df.drop_duplicates('Record ID#')

      print(total_df.columns)



   if _merge_tables_text_analysis:
      analysis = CrashAnalysis.TextAnalysis(total_df)

      ## Customer Description Analysis
      # print('total customer descriptions: ' + str(total_df['Customer_Description'].dropna(how='any').count()))
      vocab, sortedFreq = analysis.frequency('2016040014', print_output=False, top=50)

      ## Error and StackTrace Analysis
      field = 'Message'
      hist = total_df[field].fillna(' ').value_counts()
      total = sum(total_df[field].fillna(' ').value_counts()[1:])
      print(hist)
      print(total)
      #
      # ## Customer System Environment
      # print(xml_df.fillna(' ').groupby(['InstallType', 'WorkStationType', 'OperatingSystem'])['SystemTimeUTC'].count())
      # print(sum(xml_df.fillna(' ').groupby(['InstallType', 'WorkStationType', 'OperatingSystem'])['SystemTimeUTC'].count()))
      # # print(xml_df.fillna(' ').groupby('InstallType')['ExceptionAddress'].apply(lambda x: '9be7' in x))

      ## Bug 1: External component has thrown an exception.
      bug1_msgs = [hist.index[i] for i in [1, 3, 9]]
      bug1_df_list = [total_df[total_df.Message == msg].drop_duplicates('Record ID#') for msg in bug1_msgs]
      bug1_df = pd.concat(bug1_df_list, axis=0)
      # bug1_df = total_df[total_df.Message == bug1_msg].drop_duplicates('Record ID#')

      # print(bug1_df['Customer_Description'].dropna(how='any'))
      # print(bug1_df['CustNum'].value_counts())
      # print(bug1_df['StackTrace'].value_counts())

      ## Bug 2: Method not found: 'Int32 System.Runtime.InteropServices.Marshal.SizeOf(!!0)'.
      # bug2_msg = hist.index[2]
      # print(bug2_msg)
      # bug2_df = total_df[total_df.Message == bug2_msg].drop_duplicates('Record ID#')
      # print(bug2_df['Customer_Description'].dropna(how='any'))
      # print(bug2_df['CustNum'].value_counts())
      # print(bug2_df['StackTrace'].value_counts())


   if _pro_v_basic:
      pro_name = 'ProSeries - 2016'
      basic_name = 'ProSeries Basic Edition - 2016'
      print(pro_name)
      tool.frequency('2016040014', product_id=pro_name, top=50)
      if _lda:
         model = tool.lda('2016040014', product_id=pro_name, recompute=_force_recompute, num_topics=10)
         tool.print_topics(model, num_words=10)

      print()
      print(basic_name)
      tool.frequency('2016040014', product_id=basic_name, top=50)

      if _lda:
         model = tool.lda('2016040014', product_id=basic_name, recompute=_force_recompute, num_topics=10)
         tool.print_topics(model, num_words=10)

   if _total:

      vocab, sortedFreq = tool.frequency('2016040014', print_output=False, top=50)

      if _lda:
         model = tool.lda('2016040014', recompute=_force_recompute, num_topics=10)
         tool.print_topics(model, num_words=10)


      err_codes = tool.group_by_vocab(vocab, sortedFreq, tool.df, field='CrashGUID')

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
