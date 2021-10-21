#!/usr/bin/env bash
ngfs=( 128,128,128,128,128 )
epochs=( 5000 )
trys=( 1 )
conjugates=( COL )
name_data=graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_textlines # COL CELL
# name_data=graph_k10_wh0ww0jh10jw1_min0_maxwidth0.5 # ROW
models=( transformer )
data_path=/data/TableUnderstandingData/abp_small/graphs_preprocessed
test_lst=/data2/jose/corpus/tablas_DU/icdar19_abp_small/fold1.fold
train_lst=/data2/jose/corpus/tablas_DU/icdar19_abp_small/myfolds234.fold
multihead=True
if [[ "$multihead" == "False" ]]; then
    workdir=works_ABPsmall
else
    workdir=works_ABPsmall_multihead
fi

for conjugate in "${conjugates[@]}"; do
for ngf in "${ngfs[@]}"; do
for epoch in "${epochs[@]}"; do
for model in "${models[@]}"; do
for try in "${trys[@]}"; do
python main_graph.py --batch_size 128 \
--data_path ${data_path}/${name_data} \
--epochs ${epoch} --seed ${try} --work_dir ${workdir}/${conjugate}/work_graph_${conjugate}_${ngf}ngfs_base_${try}_notext_${name_data}_mish_model${model}_textlines_RPN \
--test_lst ${test_lst} \
--train_lst ${train_lst} --model ${model} --multihead ${multihead} \
--layers ${ngf} --adam_lr 0.001 --conjugate ${conjugate} --classify CELLS --show_test 500 --show_train 50 --load_model True --num_classes 13 --no-shuffle_data
# --do_prod --prod_data /data/HisClima/hyp/graphs/graphs_preprocessed/${name_data}_prod
done
done
done
done
done
ngfs=( 32,32,32 64,64,64,64,64,64,64,64 512,512,512 )
conjugates=( ROW )
for conjugate in "${conjugates[@]}"; do
for ngf in "${ngfs[@]}"; do
for epoch in "${epochs[@]}"; do
for model in "${models[@]}"; do
for try in "${trys[@]}"; do
python main_graph.py --batch_size 16 \
--data_path ${data_path}/${name_data} \
--epochs ${epoch} --seed ${try} --work_dir ${workdir}/${conjugate}/work_graph_${conjugate}_${ngf}ngfs_base_${try}_notext_${name_data}_mish_model${model}_textlines_RPN \
--test_lst ${test_lst} \
--train_lst ${train_lst} --model ${model} --multihead ${multihead} \
--layers ${ngf} --adam_lr 0.001 --conjugate ${conjugate} --classify CELLS --show_test 500 --show_train 50 --load_model True --num_classes 17
# --do_prod --prod_data /data/HisClima/hyp/graphs/graphs_preprocessed/${name_data}_prod
done
done
done
done
done