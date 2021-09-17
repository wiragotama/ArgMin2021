# preprocessing (converting tsv to vectors)
python -m preprocessing.tsv_to_vector -in_dir data/ICNALE/tsv/ -split data/ICNALE/train_test_split.csv -out_dir data/ICNALE-SBERT/

# split train set into hyperparameter tuning sets (train-x and dev); later on, we use the whole training data (train-x + dev) to train the ML models
python -m preprocessing.generate_HPT_data -in_dir data/ICNALE-SBERT/train/ -out_dir data/ICNALE-SBERT/hpt/

# check loading dataset
python -m model.datasetreader

# check whether the model is all set (test run on training)
python -m model.model_functions

# test prediction
python -m model.predict -test_dir data/ICNALE-SBERT/test/ -model_dir model/test_run/ -pred_dir model/test_pred/

# recovering UKP's missing link experiment
python -m model.predict_missing_UKP -test_dir data/UKP-SBERT/ -tsv_dir data/UKP/tsv/ -model_dir model/saved_model/MTL_CompLabel_NodeDepths_best_run_vocab_mod/ -pred_dir model/predictions/UKP_missing_links/ -save_new_tsv_dir data/UKP/tsv_recovered_links/

# evaluate prediction
python -m model.evaluate -pred_dir model/test_pred/

# train (real training)
python -m model.train -mode real_run -architecture BiaffineMTL -dir data/Multi-SmartSampling-SBERT/train/ -epochs 150 -n_run 4 -batch_size 8 -aux_tasks '["component_labels", "node_depths_labels"]' -save_dir model/saved_model/MultiData_SmartSampling_CompLabel_NodeDepths/ -save_start_no 17 -aux_features "[]"

# hyperparameter tuning
python -m model.hyperparameter_tuning -mode real_run -architecture BiaffineMTL -dir data/Multi-SmartSampling-SBERT/hpt/ -dropout_rate "[0.5]" -batch_size "[4, 8]" -epochs 300 -evaluate_every 10 -aux_tasks "['component_labels', 'node_depths_labels']" -log model/hyperparameter_tuning_log/MultiData_SmartSampling_Biaffine_CompLabel_NodeDepths_pt5.txt