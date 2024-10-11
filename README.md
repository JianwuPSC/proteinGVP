# proteinGVP
原始模型存放于https://github.com/cl666666/GVP-MSA
由seq,pdb,fitness -> 新位点变异fitness及多位点变异的fitness
1. 优点：(1) 利用了esmfold(MSA seq)+MSA transfomer(MSA seq)+GNN(pdb info), GNN_updata -> fitness
2. 缺点：(1) 可利用的样本数据少；(2) 各个模块间没有反向信息交流；(3) 多位点的话主要预测了加性效应无上位性效应；(4)  Equivariant Graph Neural Networks 可以换成 E(n) Equivariant Graph Neural Networks
3. python GVP_protein_main.py --path_prefix /home/wuj/data/protein_design/GVP_protein/params/directed_evolution_input_all_datasets/ --sample_names 'B3VI55_LIPSTSTABLE' 'BG_STRSQ' 'PTEN' 'AMIE_acet' 'HSP90' 'KKA2_KLEPN_KAN18' 'GB1_2combo' 'YAP1_WW1' 'AVGFP' 'FOS_JUN' 'TEM1' --dataset_config=/home/wuj/data/protein_design/GVP_protein/params/data_config.yaml --output_dir=deamnase_predict --epochs=201 --multi_model=True --high_order_train=False --test_names=deaminase --mut_prefix='_single' --mut_test_prefix='_single'
4. python GVP_protein_main.py --path_prefix=/home/wuj/data/protein_design/GVP_protein/params/directed_evolution_input_all_datasets --sample_names 'HSP90' 'TEM1' --dataset_config=/home/wuj/data/protein_design/GVP_protein/params/data_config.yaml --output_dir=sample_HSP90 --epochs=5 --multi_model=True --high_order_train=False --test_names=HSP90 --mut_prefix='_single'
5. python GVP_protein_main.py --path_prefix=/home/wuj/data/protein_design/GVP_protein/params/directed_evolution_input_all_datasets --sample_names 'FOS_JUN' 'AVGFP' --dataset_config=/home/wuj/data/protein_design/GVP_protein/params/data_config.yaml --output_dir=sample_HSP90 --epochs=100 --multi_model=True --high_order_train=True --test_names=TEM1 --mut_prefix='_single'
