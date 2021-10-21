import os, sys, glob, numpy as np, shutil, re
import os, sys, inspect, glob
import pickle
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from data import page

path = "/data/HisClima/hyp_newmethod/page/dla_new_method_test"
res_path = "/data/HisClima/hyp_newmethod/page/dla_new_method_test/hypotesisCoords"

files = glob.glob(os.path.join(path, "*.xml"))

file_write = open(res_path, "w")
for file in files:
    print(file)
    fname = file.split("/")[-1].split(".")[0]
    page_xml = page.TablePAGE(file)
    tls = page_xml.get_textLinesWithId()
    for coords, text, id in tls:
        text = text.replace("!print", "").replace("!manuscript", "").replace("!decimal", ".")
        coords_2 = []
        for i,j in coords:
            # print(x, type(x))
            # for i,j in x:
            coords_2.append(f'{str(i)},{str(j)}')
        # c = " ".join([f'({str(i)},{str(j)})'  for x in coords for (i,j) in x])
        # print(c)
        # exit()
        c = " ".join(coords_2)
        str_print = f'{fname}.{id} {text} Coords:( {c} )'
        file_write.write(f'{str_print}\n')
file_write.close()