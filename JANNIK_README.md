Run command:
python training.py --path_for_training_data=/ibmm_scratch/jgut/ProteinMPNN_data/pdb_2021aug02 --path_for_outputs=/ibmm_scratch/jgut/ProteinMPNN_models/default --num_epochs=200 --save_model_every_n_epochs=10 --reload_data_every_n_epochs=2 --num_examples_per_epoch=1000000 --batch_size=10000 --max_protein_length=10000 --hidden_dim=128 --num_encoder_layers=3 --num_decoder_layers=3 --num_neighbors=48 --dropout=0.1 --backbone_noise=0.2 --rescut=3.5 --gradient_norm=1 --mixed_precision=True --alpha 1


[Multimargin loss](https://arxiv.org/pdf/2501.12191) vs. [ordinal log-loss](https://aclanthology.org/2022.coling-1.407.pdf)