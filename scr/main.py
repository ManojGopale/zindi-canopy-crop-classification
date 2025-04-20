##File to run lightgbm on data
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

import numpy as np
import rasterio
from rasterio.transform import from_origin
DATA_DIR = Path('../data')
os.listdir(DATA_DIR)

# Load files
train = pd.read_csv(DATA_DIR / 'Train.csv')
test = pd.read_csv(DATA_DIR / 'Test.csv')
sample_submission = pd.read_csv(DATA_DIR / 'SampleSubmission.csv')

df = train.dropna().groupby('ID').apply(lambda x: len(pd.unique(x['Target'])))
df[df>1] ##Check if there are two targets for same ID
##Each ID has only 1 target.

##Columns to process
process_columns = test.columns
process_columns = process_columns.to_list()
##remove inplace
process_columns.remove('time') ##Remove time column for processing
process_columns.remove('ID') ##Remove ID column for processing

#train['Target'].dropna().value_counts()
#Target
#0.0    3503433
#2.0    2520826
#1.0    1532714

##Create train and validation set
x_train, x_dev, y_train, y_dev = train_test_split(
																	train.dropna()[process_columns],
																	train.dropna()['Target'],
																	test_size=0.15,
																	random_state=743,
																	) 
##Check label counts
collections.Counter(y_train)
collections.Counter(y_dev)

params = {
	'objective': 'multiclass',
	'num_class': 3,
	'metric': 'multi_logloss',
	'early_stopping_rounds': 10,
	'verbose': 1
}

#train_data = lgb.Dataset(train.dropna()[process_columns], label=train.dropna()['Target'])
train_data = lgb.Dataset(x_train, label=y_train)
dev_data = lgb.Dataset(x_dev, label=y_dev, reference=train_data)

##Check number of threads
print(f'Threads= {os.environ.get("OMP_NUM_THREADS", None)}')

num_round = 1000
model = lgb.train(params,
							train_data,
							valid_sets=[dev_data],
							num_boost_round=num_round,
							)

y_pred = model.predict(x_dev, num_iteration=model.best_iteration)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_pred, y_dev)

df_test = test.dropna()
#test_data = lgb.Dataset(df_test[process_columns])
y_test = model.predict(df_test[process_columns], num_iteration=model.best_iteration)

y_test = np.argmax(y_test, axis=1)

##https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py#L82-L84
##Save model
MODEL_DIR = Path('../model')
os.listdir(MODEL_DIR)

model.save_model(MODEL_DIR / 'model_03252025_v1.txt')

# load model to predict
bst = lgb.Booster(model_file=MODEL_DIR / 'model_03252025_v1.txt')
# can only predict with the best iteration (or the saving iteration)
y_pred_1 = bst.predict(df_test[process_columns])

df_test['Pred'] = y_test


def getMaxVote(row):
	defaultKey = 2
	sort_dict = dict(sorted(collections.Counter(row).items(), key=operator.itemgetter(1), reverse=True))
	count_list = list(sort_dict.values())
	key_list = list(sort_dict.keys())
	if(len(count_list) == 1):
		##If only 1 key is predcited
		maxKey = key_list[0]
	elif(count_list[0] > count_list[1]):
		##1st key in sorted dict has maximum count
		maxKey = key_list[0]
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


##reset_index to change ID which is index to a column.
df_submission = df_test.groupby('ID').apply(lambda row: getMaxVote(row['Pred'])).to_frame().reset_index()
df_submission.columns = sample_submission.columns

##Save output
OUTPUT_DIR = Path('../output')
os.listdir(OUTPUT_DIR)

df_submission.to_csv(OUTPUT_DIR / 'Submission.csv', index = False)
df_submission.to_csv(OUTPUT_DIR / 'model_03252025_v1_defaultkey_1_sub.csv', index = False)



######------------------------Analysis-----------------------------------#####
#model_03252025_v1_sub.csv -> 0.896953834
#model_03252025_v1_defaultkey_1_sub.csv -> 0.900004586
#model_03252025_v1_defaultkey_2_sub.csv -> 0.900004586
######-----------------------------------------------------------#####
######-----------------------------------------------------------#####
######-----------------------------------------------------------#####
######-----------------------------------------------------------#####
######-----------------------------------------------------------#####
######-----------------------------------------------------------#####
######-----------------------------------------------------------#####
######-----------------------------------------------------------#####
######-----------------------------------------------------------#####
######-----------------------------------------------------------#####
######-----------------------------------------------------------#####
