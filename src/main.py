from __future__ import print_function
import pandas as pd
import crash_analysis
from analysis import stem_frequency, associate_by_keyterms
from crash_analysis.lda import lda, print_topics


def main():
    pd.set_option('display.height', 5000)
    pd.set_option('display.max_rows', 5000)
    pd.options.display.max_colwidth = 250

    _crash_file = 'C:\\Dev\\crash_analysis\\src\\data\\Crashes3.csv'
    _merged_dataframe_file = './data/merged_df.pkl'
    _pro_v_basic = False
    _total = False
    _lda = False
    _force_recompute = False
    _extract_zipfiles = False
    _parse_xml = True
    _merge_tables = False  # dependent on parse_xml
    _merge_tables_text_analysis = True

    tool = crash_analysis.TextAnalysis(_crash_file)

    if _extract_zipfiles:
        xml_parser = crash_analysis.crash_report_parser()
        xml_parser.extract_zipfiles('C:\\CrashReports\\')

    if _parse_xml:
        xml_parser = crash_analysis.crash_report_parser()
        xml_df = xml_parser.xmldocs_to_dataframe('C:\\CrashReports\\')

    if _merge_tables:
        xml_df['CrashGUID'] = xml_df['CrashGUID'].apply(lambda x: str(x) + '.zip')
        oldcols = list(tool.df.columns)
        oldcols[5] = 'CrashGUID'
        tool.df.columns = oldcols

        print(oldcols)
        print(list(xml_df.columns))

        print('intersection: ', set(oldcols) & set(xml_df.columns))

        total_df = pd.merge(xml_df, tool.df, on='CrashGUID')
        total_df = total_df.drop_duplicates('Record ID#', keep='last')

        total_df.to_pickle(_merged_dataframe_file)

        print(total_df.columns)

    if not _merge_tables:
        total_df = pd.read_pickle(_merged_dataframe_file)


    if _merge_tables_text_analysis:
        analysis = crash_analysis.TextAnalysis(total_df)

        ## Customer Description Analysis
        # print('total customer descriptions: ' + str(total_df['Customer_Description'].dropna(how='any').count()))
        vocab, sortedFreq = stem_frequency(None, None, print_output=False, top=50)

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


        err_codes, term_count_map = associate_by_keyterms(total_df, None, field='StackTrace')

        print('Field by Keyterm')
        for word, count in sortedFreq:

            if count > 40:
                continue

            print('keyterm: ' + word)
            print('appeard in {0} rows'.format(term_count_map[word]))
            print(err_codes[word][err_codes[word] > 0])


            if count < 34:
                break

    if _pro_v_basic:
        pro_name = 'ProSeries - 2016'
        basic_name = 'ProSeries Basic Edition - 2016'
        print(pro_name)
        stem_frequency(None, None, top=50)
        if _lda:
            model = lda('2016040014', product_id=pro_name, recompute=_force_recompute, num_topics=10)
            print_topics(model, num_words=10)

        print()
        print(basic_name)
        stem_frequency(None, None, top=50)

        if _lda:
            model = lda('2016040014', product_id=basic_name, recompute=_force_recompute, num_topics=10)
            print_topics(model, num_words=10)

    if _total:

        vocab, sortedFreq = stem_frequency(None, None, print_output=False, top=50)

        if _lda:
            model = lda('2016040014', recompute=_force_recompute, num_topics=10)
            print_topics(model, num_words=10)

        associate_by_keyterms(tool.df, 'CustomerDescription', field='StackTrace')


if __name__ == '__main__':
    main()
