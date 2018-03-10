from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from time import time


start_time = time()


def file_read():
    file_data = pd.read_csv('test.txt', header=None,
                            names=['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer', 'Enjoy'],
                            skiprows=lambda x: x < 2, skipinitialspace=True)

    file_data['Occupied'] = file_data['Occupied'].str.split(' ', expand=True)[1]
    file_data['Enjoy'] = file_data['Enjoy'].str.split(';', expand=True)[0]

    return file_data.loc[:, file_data.columns != 'Enjoy'], file_data['Enjoy']


# ONE HOT ENCODING USING PANDAS

clf = tree.DecisionTreeClassifier()
data, result = file_read()
print(data.head())
new_data = pd.get_dummies(data)
print(new_data.head())
# new_result = pd.get_dummies(result)
# clf.fit(new_data, new_result)
#
# test_data = pd.DataFrame([[0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]],
#                          columns=['Occupied_High', 'Occupied_Low', 'Occupied_Moderate', 'Price_Cheap',
#                                   'Price_Expensive', 'Price_Normal',
#                                   'Music_Loud', 'Music_Quiet', 'Location_City-Center', 'Location_Ein-Karem',
#                                   'Location_German-Colony',
#                                   'Location_Mahane-Yehuda', 'Location_Talpiot', 'VIP_No', 'VIP_Yes', 'Favorite Beer_No',
#                                   'Favorite Beer_Yes'])
# new_test_data = pd.get_dummies(test_data)
#
# print(clf.predict(new_test_data))
# print(time() - start_time)

######################################################################

test = pd.DataFrame([['Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']],
                         columns=['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer'])
train_data_size = len(data)
dataset = pd.concat(objs=[data, test], axis=0)
dataset_preprocessed = pd.get_dummies(dataset)
train_data_preprocessed = dataset_preprocessed[:train_data_size]
test_preprocessed = dataset_preprocessed[train_data_size:]
print(test_preprocessed)

#####################################################################
LABEL ENCODER
le_data = LabelEncoder()
new_data = data.apply(le_data.fit_transform)
new_result = LabelEncoder().fit_transform(result)
clf.fit(new_data, new_result)

test_data = pd.DataFrame([[2, 0, 0, 0, 0, 0]],
                         columns=['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer'])
print(clf.predict(test_data))

#####################################################################

# http://what-when-how.com/artificial-intelligence/decision-tree-applications-for-data-modelling-artificial-intelligence/