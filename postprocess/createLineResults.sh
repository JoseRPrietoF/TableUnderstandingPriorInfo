hyppath=/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/COL/work_graph_COL_128,128ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_textlines_mish_modeltransformer_textlines_RPN/
data_pkls=/data2/jose/projects/RPN_LSTM/works/work_2_TextLine/results/graphs_preprocessed/graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_textlines
python createLineResults.py ${hyppath} col ${data_pkls} 0.5

# ROW
hyppath=/data2/jose/projects/TableUnderstandingPriorInfo/works_HisClima/ROW/work_graph_ROW_32,32,32ngfs_base_1_notext_graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_textlines_mish_modeltransformer_textlines_RPN/
python createLineResults.py ${hyppath} row ${data_pkls} 0.5