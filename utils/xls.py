from typing import Dict, List

import xlrd
import xlwt


class Xls:
    def __init__(self):
        return

    def read(self, xlsx_file) -> Dict[str, List[List[str]]]:
        """
        :param xlsx_file: 输入xls文件
        :return: Dict[str, List[List[str]]]: key-sheet_name value:sheet_data
        """
        workbook = xlrd.open_workbook(xlsx_file)
        sheet_names = workbook.sheet_names()
        sheet_data_dict: Dict[str, List[List[str]]] = dict()
        for sheet_name in sheet_names:
            worksheet = workbook.sheet_by_name(sheet_name)
            sheet_data: List[List[str]] = list()
            for row in range(worksheet.nrows):
                data_list: List[str] = list()
                for col in range(worksheet.ncols):
                    cell_value = worksheet.cell_value(row, col)
                    data_list.append(cell_value)
                sheet_data.append(data_list)
            sheet_data_dict[sheet_name] = sheet_data
        return sheet_data_dict

    def write(self, sheet_dict: Dict[str, List[List[str]]], xlsx_file: str):
        """
        :param sheet_dict: Dict[str, List[List[str]]]: key-sheet_name value:sheet_data
        :param xlsx_file: 需要保存的xls文件
        :return:
        """
        workbook = xlwt.Workbook()
        for sheet_name, sheet_data in sheet_dict.items():
            worksheet = workbook.add_sheet(sheet_name)
            for row, row_data in enumerate(sheet_data):
                for col, data in enumerate(row_data):
                    worksheet.write(row, col, data)
        workbook.save(xlsx_file)
        return
