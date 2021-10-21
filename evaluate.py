from utils.metrics import evaluate_graph_IoU, read_results
import os, glob, sys

def read_list_files(p, path_files):
    files = glob.glob(os.path.join(path_files, "*pkl"))
    f = open(p, "r")
    l = f.readlines()
    f.close()
    lines = [x.strip().split(".")[0] for x in l]
    res = []
    for line in lines:
        for f in files:
            if line in f:
                res.append(f)
    return res

def read_results_cells(fname_cols, fname_rows):
    """
    Since the differents methods tried save results in different formats,
    we try to load all possible formats.
    """
    results = {}
    f = open(fname_cols, "r")
    lines = f.readlines()
    f.close()
    for line in lines[1:]:
        id_line, label, prediction = line.split(" ")
        id_line = id_line.split("/")[-1].split(".")[0]
        results[id_line] = (int(label), np.exp(float(prediction.rstrip())) )
    
    f = open(fname_rows, "r")
    lines = f.readlines()
    f.close()
    for line in lines[1:]:
        id_line, label, prediction = line.split(" ")
        id_line = id_line.split("/")[-1].split(".")[0]
        results[id_line] = (int(label), np.exp(float(prediction.rstrip())) )

    return results

def main(path_files, name_fold, fresults, type_="col", conjugate=False, all_edges = True):
    """test_dataset.ids, labels, predictions"""
    list_te = read_list_files(name_fold, path_files)
    print(len(list_te))
    min_w = 0.5
    # fP, fR, fF, res = evaluate_graph_IoU(list_te, fresults, min_w=min_w, th=0.8, type_=type_, conjugate=conjugate, all_edges=all_edges)
    fP_1, fR_1, fF_1, res_1 = evaluate_graph_IoU(list_te, fresults, min_w=min_w, th=1.0, type_=type_, conjugate=conjugate, all_edges=all_edges)
    print("#####   IoU and alignment of connected components  #####")
    # print("Mean Precision IoU th 0.8 on test : {}".format(fP))
    # print("Mean Recal IoU th 0.8 on test : {}".format(fR))
    # print("Mean F1 IoU th 0.8 on test : {}".format(fF))
    print("Mean Precision IoU th 1.0 on test : {}".format(fP_1))
    print("Mean Recal IoU th 1.0 on test : {}".format(fR_1))
    print("Mean F1 IoU th 1.0 on test : {}".format(fF_1))

def main_cells(path_files, name_fold, fresults_rows, fresults_cols, type_="cell", conjugate=False, all_edges = True):
    """test_dataset.ids, labels, predictions"""
    list_te = read_list_files(name_fold, path_files)
    print(len(list_te))
    min_w = 0.5
    results = read_results_cells(read_results)
    fP, fR, fF, res = evaluate_graph_IoU(list_te, results, min_w=min_w, th=0.8, type_=type_, conjugate=conjugate, all_edges=all_edges)
    fP_1, fR_1, fF_1, res_1 = evaluate_graph_IoU(list_te, results, min_w=min_w, th=1.0, type_=type_, conjugate=conjugate, all_edges=all_edges)
    print("#####   IoU and alignment of connected components  #####")
    print("Mean Precision IoU th 0.8 on test : {}".format(fP))
    print("Mean Recal IoU th 0.8 on test : {}".format(fR))
    print("Mean F1 IoU th 0.8 on test : {}".format(fF))
    print("Mean Precision IoU th 1.0 on test : {}".format(fP_1))
    print("Mean Recal IoU th 1.0 on test : {}".format(fR_1))
    print("Mean F1 IoU th 1.0 on test : {}".format(fF_1))

if __name__ == "__main__":
    conjugate=True
    all_edges=False
    # type_="row"
    name_fold = "/data/HisClima/DatosHisclima/all.lst"
    path_files = sys.argv[1]
    type_ = sys.argv[2] 
    if type_ == "col":
        fresults = os.path.join(path_files, "results_cols.txt")
    if type_ == "span":
        fresults = os.path.join(path_files, "results_span.txt")
    elif type_ == "row":
        fresults = os.path.join(path_files, "results_rows.txt")
    main(path_files, name_fold, fresults, type_, conjugate=conjugate, all_edges=all_edges)

    # fresults = "/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_structure_BL_maxBLWidth1.5_multpunct_w4_h1_mult_side_h0_w0_bound0_rows9/results_rows.txt"
    # fresults = "/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_structure_BL_maxBLWidth1.5_multpunct_w4_h1_mult_side_h0_w0_bound0_rows9/results_rows.txt"
    # main_cells(path_files, name_fold, fresults_rows, fresults_cells, type_, conjugate=conjugate, all_edges=all_edges)
