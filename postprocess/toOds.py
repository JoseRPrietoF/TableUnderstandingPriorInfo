from pyexcel_ods3 import save_data
from collections import OrderedDict
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
try:
    from data.page import TablePAGE
except:
    from page import TablePAGE

def read_file(p:str):
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = {}
    for line in lines[1:]:
        line = line.strip()
        idline ,col, row, header = line.split(" ")
        col, row = int(col), int(row)
        page = "_".join(idline.split("_")[:-1])
        lines_page = res.get(page, [])
        lines_page.append([idline ,col, row, header])
        res[page] = lines_page
    return res



GT = True
path = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/rsults_hyp/NER_exp1{}/estructura"
if GT:
    path = path.format("_GT")
else:
    path = path.format("")
path_pages = "/data/HisClima/hyp_newmethod/page/dla_new_method_test/"
save_path = "NER_exp1"
if not os.path.exists(save_path):
    os.mkdir(save_path)
res = read_file(path)
# print(res)
data = OrderedDict()
# data.update({"Sheet 1": [[1, 2, 3], [4, 5, 6]]})
# data.update({"Sheet 2": [["row 1", "row 2", "row 3"]]})
# save_data("your_file.ods", data)
num_datos = 5
count = 0
for page, lines_page in res.items():
    count +=  1
    print(page)
    PAGE_table_path = os.path.join(path_pages, page+".xml")
    PAGE_table = TablePAGE(PAGE_table_path)
    tls_dict = PAGE_table.get_textLinesDict()
    # print(tls_dict.keys())
    text_dict = {}
    max_col, max_row = 0, 0
    for idline ,col, row, header in lines_page:
        coords, text = tls_dict.get(idline)
        text = text.replace("!print", "").replace("!manuscript", "").replace("!decimal", ".")
        # print(idline ,col, row, header, text, coords) 
        text_array, coords_array = text_dict.get((col, row), ([], []))
        text_array.append(text)
        coords_array.append(coords)
        max_col = max(max_col, col)
        max_row = max(max_row, row)
        text_dict[(col, row)] = ( text_array, coords_array)
        # if col == 0:
        #     print((row, col), text)
    # print(text_dict)
    rows = [[]]*(max_row+1)
    rows_coords = [[]]*(max_row+1)
    for i, r in enumerate(rows):
        r = [[]]*(max_col+1)
        rows[i] = r
        r2 = [[]]*(max_col+1)
        rows_coords[i] = r2
    for (col, row), (text_array, coords_array) in text_dict.items():
        if col == max_col or row == max_row or col == -1 or row == -1: 
            continue
        rows[row][col] = " ".join(text_array)
       
        coords_array = "".join([str(x) for x in coords_array])
        rows_coords[row][col] = "".join(coords_array)
        # print(col, row, text_array)

    # print(rows)
    res_ods = []
    res_ods_coords = []
    for r_i, row in enumerate(rows):
        coords_i = rows_coords[r_i]
        row_res = []
        row_res_coords = []
        for count_r, r in enumerate(row):
            coords_i_r = coords_i[count_r]
            if not r:
                r = ""
                coords_i_r = ""
            row_res.append(r)
            row_res_coords.append(coords_i_r)
        res_ods.append(row_res)
        res_ods_coords.append(row_res_coords)
    data.update({"Sheet 1": res_ods})
    data.update({"Sheet 2": res_ods_coords})
    if not GT:
        path_save_f = os.path.join(save_path, f'{page}.ods')
        
    else:
        path_save_f = os.path.join(save_path, f'{page}_GT.ods')
    save_data(path_save_f, data)
    if count >= num_datos:
        break