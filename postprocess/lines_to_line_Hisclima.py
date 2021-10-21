import glob, sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from data.page import TablePAGE
path = "/data/HisClima/hyp/pageWithText/"
path_res = "/data/HisClima/hyp/results_file"

def process_files_aux(all_files):
    res = {}
    for file in all_files:
        f = open(file, "r")
        lines = f.readlines()
        f.close
        for line in lines:
            line = line.strip()
            info, *data = line.split(" ")
            fname, _, idLine = info.split(".")
            data = " ".join(data)
            data = data.replace(" ", "").replace("!decimal", ".").replace("<print>", "").replace("<manuscript>", "").replace("<space>", " ").replace("\n", "").replace("''", "\"")
            res_fname = res.get(fname, {})
            res_fname[idLine] = data
            res[fname] = res_fname
    return res

def process_files(all_files):
    res = {}
    for file in all_files:
        page = TablePAGE(file)
        lines = page.get_textLinesWithId()
        fname = file.split("/")[-1].split(".")[0]
        for coords, text, idLine in lines:
            res_fname = res.get(fname, {})
            res_fname[idLine] = (text, coords)
            res[fname] = res_fname
    return res

def main():
    all_files = glob.glob(os.path.join(path,"*xml"))
    res = process_files(all_files)
    f = open(path_res, "w")
    for fname, lines in res.items():
        for idLine, (text, coords) in lines.items():
            coords_str = ""
            for x,y in coords:
                coords_str += f'{int(x)},{int(y)} '
            t = f'{fname}.{idLine} {text} Coords:( {coords_str})'
            # print(t)
            f.write(f'{t}\n')
            # exit()
    f.close()


if __name__ == "__main__":
    main()