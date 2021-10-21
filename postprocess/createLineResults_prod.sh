# COL
hyppath=/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/COL/work_graph_COL_128,128ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_mish_modeltransformer_3/
data_pkls=/data/HisClima/hyp/graphs/graphs_preprocessed/graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_prod
python createLineResults.py ${hyppath} col ${data_pkls} 0.5 no true

# ROW
hyppath=/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/COL/work_graph_COL_128,128ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_mish_modeltransformer_3/
data_pkls=/data/HisClima/hyp/graphs/graphs_preprocessed/graph_k10_wh0ww0jh10jw1_min0_maxwidth0.5_prod
python createLineResults.py ${hyppath} row ${data_pkls} 0.5 no true