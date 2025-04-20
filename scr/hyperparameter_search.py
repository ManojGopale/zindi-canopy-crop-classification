'''This file searches for best hyper-parameters for lightgbm
https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_integration.py
'''
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
import operator
import optuna
import pdb

import numpy as np
import rasterio
from rasterio.transform import from_origin

def getMaxVote(row):
	'''Calculates the maximum occuring number in the list
	'''
	defaultKey = 2
	# Sort the dicionary of {num: count} in descending order
	sort_dict = dict(sorted(collections.Counter(row).items(), key=operator.itemgetter(1), reverse=True))
	count_list = list(sort_dict.values())
	key_list = list(sort_dict.keys())
	if(len(count_list) == 1):
		##If only 1 key is predcited
		maxKey = key_list[0]
	elif(count_list[0] > count_list[1]):
		##1st key in sorted dict has maximum count
		maxKey = key_list[0]
	elif(count_list[0] == count_list[1] and len(count_list)==2):
		##2 values are predicted and both are equal in number
		if(defaultKey in key_list):
			maxKey = defaultKey
		else:
			maxKey = 1
	elif(count_list[0] == count_list[1] and count_list[0] > count_list[2]):
		##1st and 2nd counts are equal and is greater than last count
		##Choose lower key
		#if(0 in (key_list[0], key_list[1])):
		if(defaultKey in (key_list[0], key_list[1])):
			maxKey = defaultKey
		else:
			maxKey = 1 ##defaultKey=0, 2, Since key number 1 will be in the second tuple.
			#maxKey = 2 ##when defaultKey=1, since defaultKey=2 increases the accuracy
	else: ##if all of them are equal count
		maxKey = defaultKey ##Choose 0 by default if all have equal count
	return maxKey

def trainModel(param):
	# Load files
	train = pd.read_csv(DATA_DIR / 'Train.csv')
	test = pd.read_csv(DATA_DIR / 'Test.csv')
	sample_submission = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')
	##Columns to process
	process_columns = test.columns
	process_columns = process_columns.to_list()
	##remove inplace
	process_columns.remove('time') ##Remove time column for processing
	process_columns.remove('ID') ##Remove ID column for processing
	x_train, x_dev, y_train, y_dev = train_test_split(
																		train.dropna()[process_columns],
																		train.dropna()['Target'],
																		test_size=0.15,
																		random_state=743,
																		) 
	train_data = lgb.Dataset(x_train, label=y_train)
	dev_data = lgb.Dataset(x_dev, label=y_dev, reference=train_data)
	gbm = lgb.train(param, train_data, valid_sets=[dev_data])

	preds = gbm.predict(x_dev, num_iteration=gbm.best_iteration)
	#pdb.set_trace()
	preds = np.argmax(preds, axis=1)

	train_pred = gbm.predict(x_train, num_iteration=gbm.best_iteration)
	train_pred = np.argmax(train_pred, axis=1)

	acc_dev = accuracy_score(y_dev, preds)
	acc_train = accuracy_score(y_train, train_pred)

	return gbm, acc_train, acc_dev, process_columns

##Prediction
def predTest(gbm, df, process_columns):
	'''
	Returns dataframe with predictions. Each ID is grouped and maximum occuring prediction is returned
	'''
	pred = gbm.predict(df[process_columns], num_iteration=gbm.best_iteration)
	df['Pred'] = np.argmax(pred, axis=1)
	df = df.groupby('ID').apply(lambda row: getMaxVote(row['Pred'])).to_frame().reset_index()
	cols = ['ID', 'Target']
	df.columns = cols
	return df


def objective(trial):
	param = {
		"num_boost_round": num_round,
		"num_class": 3,
		"objective": "multiclass",
		"metric": "multi_logloss",
		"verbosity": 1,
		"early_stopping_rounds": 10,
		"boosting_type": "gbdt",
		"lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
		"lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
		"num_leaves": trial.suggest_int("num_leaves", 2, 256),
		"feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
		"bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
		"bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
		"min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
	}

	gbm, acc_train, acc_dev, _ = trainModel(param)
	##Set user defined attributes to track
	trial.set_user_attr("acc_dev", acc_dev)
	trial.set_user_attr("acc_train", acc_train)

	return acc_dev

if __name__ == "__main__":
	##Check number of threads
	print(f'Threads= {os.environ.get("OMP_NUM_THREADS", None)}')

	##Create a hyper-parameter tuning study
	num_round = 2000
	DATA_DIR = Path('../data')
	study = optuna.create_study(
			pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
			direction="maximize",
			study_name='study_03282025'
			)
	study.optimize(objective, n_trials=15)
	print("Number of finished trials: {}".format(len(study.trials)))
	print("Best trial:")
	trial = study.best_trial
	print("  Value: {}".format(trial.value))
	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))
	
	##Save the entire study dataframe
	OUTPUT_DIR = Path('../output/' + study.study_name)
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	study.trials_dataframe().to_csv(OUTPUT_DIR / 'trials.csv')

	##Train model with best params
	##study.best_params has searched parameters. They are missing the constant value parameters.
	## We need to merge the constant parameter values to get the complete list for retraining
	p1 = {"num_boost_round": num_round, "num_class": 3, "objective": "multiclass", "metric": "multi_logloss", "verbosity": 1, "early_stopping_rounds": 10, "boosting_type": "gbdt"}
	params = p1 | study.best_params ##Merge dicionaries
	gbm, acc_train, acc_dev, process_columns = trainModel(params)
	print(f"-----Best Param-----\nacc_train= {acc_train}\nacc_dev= {acc_dev}")

	##Save submission file
	test = pd.read_csv(DATA_DIR / 'Test.csv')
	df_test = test.dropna()
	df_sub = predTest(gbm, df_test, process_columns)
	df_sub.to_csv(OUTPUT_DIR / 'Submission.csv', index = False)

	#Save model
	model.save_model(OUTPUT_DIR / study.study_name + '.txt')

