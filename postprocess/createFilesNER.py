import os, sys, glob, numpy as np, shutil, re

USE_GT = False

def create_dir(path:str):
    if not os.path.exists(path):
        os.mkdir(path)

def calc_distance(coords, coords2, weights=[10,1]):
    # print(coords, coords2)
    x1 = np.mean([x[0] for x in coords])
    x2 = np.mean([x[0] for x in coords2])
    y1 = np.mean([x[1] for x in coords])
    y2 = np.mean([x[1] for x in coords2])
    dst = (((x2 - x1)*weights[0])**2 + ((y2 - y1)*weights[1])**2)**0.5
    return dst

def read_text(path_text:str, path_col:str, path_row:str, path_header:str, prod:bool=False):
    """
    return a dict with a key for page
    """
    dict_col, dict_row, dict_header = {}, {}, {}
    res = {}

    # Header
    fheader = open(path_header, "r")
    lines = fheader.readlines()
    fheader.close()
    
    for line in lines[1:]:
        line = line.strip()
        if not prod:
            lId, header_gt, *header_hyp = line.split(" ")
        else:
            lId, *header_hyp = line.split(" ")
        header_hyp = " ".join(header_hyp).strip().replace("[", "").replace("]", "")
        header_hyp = re.sub(' +', ' ', header_hyp).split(" ")
        header_hyp = [float(x) for x in header_hyp if x][-1]
        *fname, lId = lId.split("-")
        fname = "-".join(fname)
        fname = fname.split("/")[-1].split(".")[0]
        d = dict_header.get(fname, {})
        if USE_GT:
            c = int(header_gt)
        else:
            c = int(np.exp(float(header_hyp)) > 0.5)
        d[lId] = c
        dict_header[fname] = d

    dict_cols_group_header = {}
    for path_col_file in glob.glob(os.path.join(path_col, "*")):
        fname = path_col_file.split("/")[-1]
        fcol = open(path_col_file, "r")
        lines = fcol.readlines()
        fcol.close()
        d = {}
        d_header = {}
        for line in lines:
            line = line.strip()
            if not prod:
                lId, cgroup, cgroup_gt = line.split(" ")
            else:
                lId, cgroup = line.split(" ")
            if USE_GT:
                c = int(cgroup_gt)
            else:
                c = int(cgroup)
            if c == 16 or c == -1: #Es la clase para "out of table" en cols
                continue
            d[lId] = c
            is_header = dict_header[fname].get(lId, 0)
            head_group = d_header.get(d[lId], 0)
            d_header[d[lId]] = max(head_group, is_header)
        dict_col[fname] = d
        dict_cols_group_header[fname] = d_header

    for path_row_file in glob.glob(os.path.join(path_row, "*")):
        fname = path_row_file.split("/")[-1]
        frow = open(path_row_file, "r")
        lines = frow.readlines()
        frow.close()
        d = {}
        for line in lines:
            line = line.strip()
            if not prod:
                lId, cgroup, cgroup_gt = line.split(" ")
            else:
                lId, cgroup = line.split(" ")
            if USE_GT:
                c = int(cgroup_gt)
            else:
                c = int(cgroup)
            if c == 27 or c == -1: #Es la clase para "out of table" en rows
                continue
            d[lId] = c
        dict_row[fname] = d

    
    f = open(path_text, "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.strip()
        
        l, coords = line.split("Coords:")
        coords = coords.replace("( ", "").replace(" )", "")
        coords = [[int(x.split(",")[0]), int(x.split(",")[1])] for x in coords.split(" ")]

        lId, *text = l.split(" ")
        text = " ".join(text)
        page_name, lId = lId.split(".") 
        #COL
        col_group = dict_col[page_name].get(lId, -1)
        print(dict_col.keys(), page_name, lId, col_group)
        #ROW
        row_group = dict_row[page_name].get(lId, -1)
        #HEADER
        header_group = dict_header[page_name].get(lId, -1)
        info = [lId, col_group, row_group, header_group, coords, text]
        dict_page_line, dict_page_headers, dict_page_cols = res.get(page_name, [{}, {}, {}])

        header_list_page = dict_page_headers.get(header_group, [])
        header_list_page.append(info)
        dict_page_headers[header_group] = header_list_page

        cols_list_page = dict_page_cols.get(col_group, [])
        cols_list_page.append(info)
        dict_page_cols[col_group] = cols_list_page

        dict_page_line[lId] = info

        res[page_name] = [dict_page_line, dict_page_headers, dict_page_cols]
    # return res
    for fname, d in dict_col.items():
        # fname = 'vol003_118_0'
        d = dict_col[fname]
        for group, has_header in dict_cols_group_header[fname].items():
            if not has_header:
                # print(group, fname)
                # Get minium Y
                coords_arr = []
                print(res[fname][2].keys(), fname, group)
                exit()
                cols = res[fname][2][group]
                for lId, col_group, row_group, header_group, coords, text in cols:
                    coords_arr.append(coords)
                coords_arr.sort(key = lambda x: x[1])
                min_y = coords_arr[0]
                coords_arr = []
                groups = list(res[fname][2].keys())
                groups.remove(group)
                for group_ in groups:
                    cols = res[fname][2][group_]
                    for lId, col_group, row_group, header_group, coords, text in cols:
                        if col_group != -1:
                            coords_arr.append([coords, col_group])
                coords_arr.sort(key = lambda x: x[0][1], reverse=True)
                # print(min_y)
                # print(coords_arr)
                # Search minium distance with mahalanobies
                new_group, min_d = -1, 99999
                coords_subject = []
                for i, [coords, group_] in enumerate(coords_arr):
                    d = calc_distance(min_y, coords)
                    if d < min_d:
                        min_d = d
                        new_group = group_
                        coords_subject = coords
                # print(min_y, coords_subject, new_group, min_d)
                cols = res[fname][2][group]
                # print(res[fname][2][new_group])
                for lId, col_group, row_group, header_group, coords, text in cols:
                    col_group = new_group
                    info = [lId, col_group, row_group, header_group, coords, text]
                    # print(info)
                    res[fname][2][new_group].append(info)
                # print(res[fname][2][new_group])
                # print(group)
                # del res[fname][2][group]
                # exit()

        # print(dict_cols_group_header[fname])
        # print(fname, d)
    # exit()

    return res

def search_first_col(cols, row_search):
    pre = ""
    for lId, col_group, row_group, header_group, coords, text in cols:
        if text.find("A. M.") != -1:
            pre = text
        elif text.find("P. M.") != -1:
            pre = text
        if row_group == row_search:
            pre += f'{text}'
            return pre
            # and row_group == row_group
    # print("Problem")
    # return None
    return ""

def main():
    nexp = 1
    prod = False
    path_save = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/NER_hyp/"
    path_hyp_text = "/data/HisClima/DatosHisclima/NER_vero/hypotesisCoords"
    # path_hyp_text = "/data/HisClima/hyp/results_file"
    # NOT RPN - 
    # path_col = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/COL/work_graph_COL_128,128ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_mish_modeltransformer_3/results_line_col_prod"
    # path_row = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/COL/work_graph_COL_128,128ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_mish_modeltransformer_3/results_line_row_prod/"
    # path_header = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/HEADER/work_graph__64,64,64ngfs_base_1_notext_graph_k10_wh0ww0jh10jw1_min0_maxwidth0.5_mish_modeltransformer_3/results_prod.txt"
    # RPN
    path_col = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/COL/work_graph_COL_128,128ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_textlines_mish_modeltransformer_textlines_RPN/results_line_col"
    path_row = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/ROW/work_graph_ROW_32,32,32ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_textlines_mish_modeltransformer_textlines_RPN/results_line_row/"
    path_header = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/HEADER/work_graph__64,64,64ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_textlines_mish_modeltransformer_textlines_RPN/results.txt"

    name_dir = f'NER_exp{nexp}'
    if prod:
        name_dir = f'{name_dir}_prod'
    elif USE_GT:
        name_dir = f'{name_dir}_GT'
    path_save = os.path.join(path_save, name_dir)
    create_dir(path_save)
    # shutil.copyfile("PrecRec.sh", os.path.join(path_save,"PrecRec.sh"))
    
    """Info exp"""
    info_path = os.path.join(path_save, "info_exp")
    f = open(info_path, "w")
    f.write(f'path_hyp_text {path_hyp_text} \n')
    f.write(f'path_col {path_col} \n')
    f.write(f'path_row {path_row} \n')
    f.write(f'path_header {path_header} \n')
    f.close()


    
    all_info = read_text(path_text=path_hyp_text, path_col=path_col, path_row=path_row, path_header=path_header, prod=prod)
    words_header = [["Force", "WindForce"], ["Cour", "Courses"], ["Direction", "WindDirection"], ["inches", "BarometerHeight"], ["Ther", "BarometerTher"], ["Dry", "AirTemperature"], ["Wet", "BulbTemperature"], ["surface", "SeaTemperature"], ["weather", "WeatherState"], ["Clouds", "Clouds"], ["Sky", "ClearSky"]]
    for page_name, [dict_page_line, dict_page_headers, dict_page_cols] in all_info.items():
        # print(page_name)
        headers = dict_page_headers.get(1)
        headers.extend(dict_page_headers.get(0)) # TODO fallan algunos headers
        # Search for the fist col - Hour
        query_word = "Hour".upper() #TODO
        first_col_group = -1
        for lId, col_group, row_group, header_group, coords, text in headers:
            if text.find(query_word) != -1:
                first_col_group = col_group
                print(f'--> query_word - {query_word} - found in col {col_group} row {row_group} text: - {text} - for page {page_name}')
                break
        if first_col_group == -1:
            error_msg = f'Query first col word {query_word} not found for page {page_name}'
            # raise Exception(error_msg)
            print(error_msg)
            continue
        for query_word, fname_query in words_header:
            query_word = query_word.upper()
            query_save_path = os.path.join(path_save, f'{fname_query}_{page_name}.txt')
            # print(query_save_path)
            
            col_group_searched = -1
            for lId, col_group, row_group, header_group, coords, text in headers:
                if text.find(query_word) != -1:
                    col_group_searched = col_group
                    # print(f'query_word - {query_word} - found in col {col_group} row {row_group} text: - {text} - ')
                    break
            if col_group_searched == -1:
                # print(f'Query word {query_word} not found in image {page_name}')
                continue
            # else:
            #     print(f'Query word {query_word} -------------------------  found in image {page_name}')
            cols = dict_page_cols[col_group_searched]
            cols.sort(key = lambda x: x[4][0][1])
            res_groups = {} # Per row
            for lId, col_group, row_group, header_group, coords, text in cols:
                # print(f'BL {lId} col {col_group} row {row_group} header_group {header_group} text = {text}')
                listcell, header_group = res_groups.get(row_group, [[], header_group])
                listcell.append(text.strip())
                # listcell += text
                res_groups[row_group] = [listcell, header_group]
            # print("========")
            file_query_save = open(query_save_path, "w")
            last_text = ""
            for row_group, [text, headerBool] in res_groups.items():
                text = " ".join(text)
                if headerBool:
                    # print without fisr col
                    # print(text)
                    file_query_save.write(f'{text}\n')
                else:
                    # Search for the first col
                    pre = search_first_col(dict_page_cols[first_col_group], row_group)
                    if text == "\"" or text.find("\"") != -1:
                        text = f'{text} [{last_text}]'
                    else:
                        last_text = text
                    res_text = f'{pre}{text}'
                    file_query_save.write(f'{res_text}\n')
                    
                    # print(res_text)
            file_query_save.close()
            # print(res_groups)

        # exit()
        

if __name__ == "__main__":
    main()