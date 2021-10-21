#!/usr/bin/env bash
# ABP_SMALL
#COLS
# data_path(PAGE-XML) dir_dest min_cell min_num_neighrbors weight_radius_w weight_radius_h j_h h_w
# data_path="/data/HisClima/DatosHisclima/data/GT-Corregido/page/"
data_path_tablelabel="/data/HisClima/DatosHisclima/data/GT-Corregido/page"
# data_path="/home/jose/projects/RPN_LSTM/works/work_2_TextLine/results/all_notable/"
data_path="/data/HisClima/hyp_newmethod/page/dla_new_method_test/"
min_cell=0
min_num_neighrbors=10
weight_radius_w=0
weight_radius_h=0
j_h=1
j_w=1
max_width_line=0.5
type_data=textlines
prod=false
hisclima=true
info_attributes=false
name_path="k${min_num_neighrbors}_wh${weight_radius_h}ww${weight_radius_w}jh${j_h}jw${j_w}_min${min_cell}_maxwidth${max_width_line}_${type_data}"
# dir_dest=/data2/jose/projects/RPN_LSTM/works/work_2_TextLine/results/graphs/graph_${name_path}
# dir_dest_processed=/data2/jose/projects/RPN_LSTM/works/work_2_TextLine/results/graphs_preprocessed/graph_${name_path}
dir_dest=/data/HisClima/hyp_newmethod/page/graphs_dla_new_method/graph_${name_path}
dir_dest_processed=/data/HisClima/hyp_newmethod/page/graphs_dla_new_method/graphs_preprocessed/graph_${name_path}
cd /data2/jose/projects/TableUnderstandingPriorInfo/data
python create_graphs.py ${data_path} ${dir_dest} ${min_cell} ${min_num_neighrbors} ${weight_radius_w} ${weight_radius_h} ${j_h} ${j_w} ${max_width_line} ${hisclima} ${prod} ${type_data} ${info_attributes} ${data_path_tablelabel}
python preprocess.py ${dir_dest} ${dir_dest_processed} false false 
python create_results.py ${dir_dest_processed} cols
python create_results.py ${dir_dest_processed} rows
# python create_results.py ${dir_dest_processed} row 
# cd /data2/jose/projects/TableUnderstandingPriorInfo/
# python3.6 evaluate.py ${dir_dest}
# cd /data2/jose/projects/TableUnderstandingPriorInfo/postprocess
# python3.6 connected_components.py ${dir_dest}
# cd /data2/jose/projects/TableUnderstandingPriorInfo/data
echo ${name_path}
#graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5 COL 
#graph_k10_wh0ww0jh10jw1_min0_maxwidth0.5 ROW