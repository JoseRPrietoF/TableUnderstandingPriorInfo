import os, sys, glob, numpy as np, shutil, re

USE_GT = True

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

def read_text(path_col:str, path_row:str, path_header:str, prod:bool=False):
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
            # if c == 16 or c == -1: #Es la clase para "out of table" en cols
            #     continue
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
            # if c == 27 or c == -1: #Es la clase para "out of table" en rows
            #     continue
            d[lId] = c
        dict_row[fname] = d

    print(len(dict_col), len(dict_row), len(dict_header))
    for page, values_col in dict_col.items():
        values_row = dict_row[page]
        values_header = dict_header[page]
        # print(len(values_col), len(values_row), len(values_header))
        for line, value_col in values_col.items():
            value_row = values_row[line]
            value_header = values_header[line]
            # print(line, value_col, value_row, value_header)
            res[line] = (value_col, value_row, value_header)
        # exit()
    
   
    return res


def main():
    nexp = 1
    prod = False
    path_save = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/rsults_hyp/"
    # path_hyp_text = "/data/HisClima/hyp/results_file"
    # NOT RPN - 
    # path_col = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/COL/work_graph_COL_128,128ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_mish_modeltransformer_3/results_line_col_prod"
    # path_row = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/COL/work_graph_COL_128,128ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_mish_modeltransformer_3/results_line_row_prod/"
    # path_header = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/HEADER/work_graph__64,64,64ngfs_base_1_notext_graph_k10_wh0ww0jh10jw1_min0_maxwidth0.5_mish_modeltransformer_3/results_prod.txt"
    # RPN
    path_col = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/COL/work_graph_COL_128,128ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_textlines_mish_modeltransformer_textlines_RPN_prev/results_line_col"
    path_row = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/ROW/work_graph_ROW_32,32,32ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_textlines_mish_modeltransformer_textlines_RPN_prev/results_line_row/"
    path_header = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/HEADER/work_graph__64,64,64ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_textlines_mish_modeltransformer_textlines_RPN_prev/results.txt"

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
    f.write(f'path_col {path_col} \n')
    f.write(f'path_row {path_row} \n')
    f.write(f'path_header {path_header} \n')
    f.close()

    all_info = read_text(path_col=path_col, path_row=path_row, path_header=path_header, prod=prod)
    res_path = os.path.join(path_save, "estructura")
    f = open(res_path, "w")
    f.write(f'idline col row header\n')
    for line, (value_col, value_row, value_header) in all_info.items():
        f.write(f'{line} {value_col} {value_row} {value_header}\n')
    f.close()
    print(f'Un total de {len(all_info)} lineas han sido procesadas y guardadas en {res_path}')
        

if __name__ == "__main__":
    main()