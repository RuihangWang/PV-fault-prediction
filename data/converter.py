
import csv
import xlrd
import glob
import pandas as pd


def csv_from_excel(xlsx_file, sheet_name, output_file):
    wb = xlrd.open_workbook(xlsx_file)
    sh = wb.sheet_by_name(sheet_name)
    your_csv_file = open(output_file, 'a')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(1, sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()


def merge_csv():
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.to_csv("All_data.csv", index=False, encoding='utf-8-sig')


# runs the csv_from_excel function:
if __name__ == '__main__':
    # csv_from_excel(xlsx_file='PVArray10x9.xlsx', sheet_name='SC_S2P8', output_file='PV_Data_10x9.csv')
    merge_csv()

