import numpy as np, os
from sklearn.metrics import confusion_matrix

def read_results(fname):
    results = {}
    if type(fname) == str: 
        f = open(fname, "r")
        lines = f.readlines()
        f.close()
        ls = []
        text = ""
        for line in lines[1:]:
            line = line.rstrip()
            if line.startswith("/") :
                ls.append(text)
                text = line
                    
            else:
                text += " " + line
        ls.append(text)
        ls = ls[1:]
        res = []
        for line in ls:
            # if "vol003_150_0.xml" not in line:
            #     continue
            # print(line)
            id_line, label, *prediction = line.split(" ")
            id_line = id_line.split("/")[-1]#.split(".")[0]
            # prediction = ""
            # p = line.split("[")[-1].replace("]", "")
            # print(line.split("["))
            prediction = [x.replace("[", "").replace("]", "") for x in prediction if x]
            prediction = [x for x in prediction if x != ""]
            p = [float(x) for x in " ".join(prediction).rstrip().split(" ")]
            p = np.exp(p)
            p = np.argmax(p)   
            
            res.append(p == int(label))         
            # print(p)
            results[id_line] = (int(label), p )
    else:
        # print(fname)
        for id_line, label, prediction in fname:
            id_line = id_line.split("/")[-1].split(".")[0]
            p = np.exp(float(prediction))
            results[id_line] = int(label), p
    return results

def main(path):
    results = read_results(os.path.join(path, "results.txt"))
    y_true, y_pred = [], []
    for k,v in results.items():
        gt, hyp = v
        y_true.append(gt)
        y_pred.append(hyp)
    conf_matr = confusion_matrix(y_true, y_pred)
    save_path_conf = os.path.join(path, "confmat.txt")
    print(conf_matr)
    f = open(save_path_conf, "w")
    for row in conf_matr:
        for col in row:
            f.write(f'{col}\t')
        f.write("\n")
    f.close()

if __name__ == "__main__":
    # path = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/COL/work_graph_COL_128,128ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_mish_modeltransformer_3/"
    path = "/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/ROW/work_graph_ROW_32,32,32ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_mish_modeltransformer_3/"
    main(path)