import os, sys, inspect, glob
import pickle
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from data import page

def read_htr(htr_path):
    all_hyp = glob.glob(os.path.join(htr_path, "*_latgen.hyp"))
    res = dict()
    for file in all_hyp:
        f = open(file, "r")
        lines = f.readlines()
        f.close()
        for line in lines:
            file = line.strip()
            info = line.split(" ")[0]
            data = line.split(" ")[1:]
            fname, _, idLine = info.split(".")
            data = " ".join(data)
            data = data.replace(" ", "").replace("<decimal>", ".").replace("<print>", "").replace("<manuscript>", "").replace("<space>", " ").replace("\n", "")
            lineas_file = res.get(fname, {})
            lineas_file[idLine] = data
            res[fname] = lineas_file
            print(fname, idLine, data)
            # exit()
    return res

def main(args):
    htr_path = "/data/HisClima/hyp/results"
    pages = "/data/HisClima/hyp/page"
    lines = read_htr(htr_path)
    # print(lines)
    for fname, lines_data in lines.items():
        fpath = os.path.join(pages, f'{fname}.xml')
        pagefile = page.TablePAGE(fpath)
        for idLine, text in lines_data.items():
            pagefile.set_text(idLine, text)
        pagefile.save_changes()
        print(fpath)
        # exit()


if __name__ == "__main__":
    main(sys.argv)