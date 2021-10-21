#!/usr/bin/env bash
# ABP_SMALL
#COLS
# data_path(PAGE-XML) dir_dest min_cell min_num_neighrbors weight_radius_w weight_radius_h j_h h_w
data_path="/data/HisClima/DatosHisclima/data/test_pages/"
min_cell=0
min_num_neighrbors=10
weight_radius_w=0
weight_radius_h=0
j_h=1
j_w=1
max_width_line=0.5
name_path="k${min_num_neighrbors}_wh${weight_radius_h}ww${weight_radius_w}jh${j_h}jw${j_w}_min${min_cell}_maxwidth${max_width_line}"
dir_dest=/data/HisClima/DatosHisclima/graphs/graph_${name_path}
dir_dest_processed=/data/HisClima/DatosHisclima/graphs/graphs_preprocessed/graph_${name_path}
cd /data2/jose/projects/TableUnderstandingPriorInfo/data
python3.6 EDA.py ${data_path} ${dir_dest} ${min_cell} ${min_num_neighrbors} ${weight_radius_w} ${weight_radius_h} ${j_h} ${j_w} ${max_width_line} true