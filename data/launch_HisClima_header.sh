#!/usr/bin/env bash
# ABP_SMALL
#COLS
# data_path(PAGE-XML) dir_dest min_cell min_num_neighrbors weight_radius_w weight_radius_h j_h h_w
data_path="/data/HisClima/DatosHisclima/data/GT-Corregido/page/"
min_cell=0.3 # multiplica al radio minimo "min_radio"
min_num_neighrbors=10
weight_radius_w=0.5
weight_radius_h=0.5
j_h=1
j_w=1
max_width_line=0.5
name_path="k${min_num_neighrbors}_wh${weight_radius_h}ww${weight_radius_w}jh${j_h}jw${j_w}_min${min_cell}_maxwidth${max_width_line}"
dir_dest=/data/HisClima/DatosHisclima/graphs/graph_${name_path}_headers
dir_dest_processed=/data/HisClima/DatosHisclima/graphs/graphs_preprocessed/graph_${name_path}_headers
cd /data2/jose/projects/TableUnderstanding/data
# python3.6 create_graphs_headers.py ${data_path} ${dir_dest} ${min_cell} ${min_num_neighrbors} ${weight_radius_w} ${weight_radius_h} ${j_h} ${j_w} ${max_width_line} true
# python3.6 preprocess.py ${dir_dest} ${dir_dest_processed} True
# python3.6 create_results.py ${dir_dest_processed} span
cd /data2/jose/projects/TableUnderstanding/
# python3.6 evaluate.py ${dir_dest_processed} span
cd /data2/jose/projects/TableUnderstanding/postprocess
python3.6 connected_components.py ${dir_dest_processed} span si
cd /data2/jose/projects/TableUnderstanding/data
echo ${dir_dest_processed}