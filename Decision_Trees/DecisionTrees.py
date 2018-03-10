from math import log2
import pandas as pd
import numpy as np
from time import time

start_time = time()


# Node for Decision Tree
class Node:

    def __init__(self, attribute_name):
        self.attribute_name = attribute_name
        self.children = dict()

    def addChild(self, childNode, category):
        self.children.update({category:childNode})


# To store the precedence of the attributes as mentioned in the question document.
attribute_order = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer']


# Calculate all possible values of attributes, so branches for each value can be created.
def calculate_unique_values_of_attributes(data):
    result = dict()
    for att in attribute_order:
        result[att] = data[att].unique()

    return result


# Calculate the most common class in the data given. Used in cases when there
# is an attribute combination which is not present in training data.
def calculate_most_common_class(data):
    data_count = data.value_counts()
    yes_count = data_count['Yes']
    no_count = data_count['No']
    if yes_count > no_count:
        return 'Yes'
    return 'No'


def file_read():
    file_data = pd.read_csv('dt-data.txt', header=None,
                            names=['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer', 'Enjoy'],
                            skiprows=lambda x: x < 2, skipinitialspace=True)

    # To clean the file data and remove unnecessary characters
    file_data['Occupied'] = file_data['Occupied'].str.split(' ', expand=True)[1]
    file_data['Enjoy'] = file_data['Enjoy'].str.split(';', expand=True)[0]

    return file_data


# Calculate Entropy given the yes and no count for that attribute
def calculateEntropy(yes_count, no_count):
    if yes_count == 0 or no_count == 0:
        return 0
    else:
        total_count = yes_count + no_count
        p = yes_count / total_count
        entropy = -1 * p * log2(p) - (1 - p) * log2(1 - p)
        return entropy


# Calculate weighted entropy
def calc_weighted_entropy(attribute_column, result_column):

    total_entropy = 0
    for value in set(attribute_column):
        yes_count = no_count = 0
        for i in range(len(attribute_column)):
            if attribute_column[i] == value:
                if result_column[i] == 'Yes':
                    yes_count += 1
                else:
                    no_count += 1
        total_entropy += (calculateEntropy(yes_count, no_count) * (yes_count+no_count)) / len(attribute_column)

    return total_entropy


# Calculate entropy at the start/root
def calc_initial_entropy(result_data):
    if len(set(result_data)) == 1:
        return 0
    else:
        data_count = result_data.value_counts()
        yes_count = data_count['Yes']
        no_count = data_count['No']
        return calculateEntropy(yes_count, no_count)


# Helper method to form a new list given a column from dataframe.
# This is used since on slicing from dataframe the indexes are irregular.
# This would allow having continuous indexes
def get_new_list(column):
    result = []
    for value in column:
        result.append(value)

    return result


# Main recursive function
def generateTree(data, attribute_list, parent_entropy):
    unique_values_of_output = data['Enjoy'].unique()
    max_info_gain = 0

    if len(unique_values_of_output) == 1:
        return Node(unique_values_of_output[0])
    elif not attribute_list:
        unique_elements = np.unique(data['Enjoy'])
        return Node(max(unique_elements))
    else:
        selected_attribute = ''
        for i in range(len(attribute_list)):
            attribute_column = get_new_list(data[attribute_list[i]])
            result_column = get_new_list(data['Enjoy'])
            entropy = calc_weighted_entropy(attribute_column, result_column)
            info_gain = parent_entropy - entropy
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                selected_attribute = attribute_list[i]
            elif info_gain == max_info_gain:
                if selected_attribute == '':
                    selected_attribute = attribute_list[i]
                else:
                    early_index = min(attribute_order.index(selected_attribute), attribute_order.index(attribute_list[i]))
                    selected_attribute = attribute_order[early_index]

        new_node = Node(selected_attribute)
        for value in unique_attr_value_dict[selected_attribute]:
            sliced_data = data.loc[data[selected_attribute] == value].drop(selected_attribute, axis=1)
            if sliced_data.empty:
                new_node.addChild(Node(calculate_most_common_class(data['Enjoy'])), value)
            else:
                upper_entropy = calc_initial_entropy(sliced_data['Enjoy'])
                updated_attr_list = [x for x in attribute_list if x != selected_attribute]
                temp_node = generateTree(sliced_data, updated_attr_list, upper_entropy)
                new_node.addChild(temp_node, value)

        return new_node


# Method to write the tree structure in the given format
def write(root_node, level):
    print(' ', root_node.attribute_name)
    for key, value in root_node.children.items():
        print('    ' * (level+1), key, ':', end="")
        write(value, level+1)


# To pass data and test the decision tree
def test(node, data):
    while node:
        if node.attribute_name == 'Yes' or node.attribute_name == 'No':
            return node.attribute_name
        else:
            attribute = node.attribute_name
            value = data[attribute][0]
            node = node.children[value]


data = file_read()
unique_attr_value_dict = calculate_unique_values_of_attributes(data)
init_entropy = calc_initial_entropy(data['Enjoy'])
root = generateTree(data, attribute_order, init_entropy)
write(root, 1)
test_data = pd.DataFrame([['Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']],
                         columns=['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer'])

print('Output : ', test(root, test_data))

# To calculate time the code takes to run
# print(time() - start_time)