import time     # required in several code segments for keeping count of time
# default = 0.01
chi_square_threshold = 6.635

# class to help structure the nodes in tree
class TreeNode:
    value = ""
    successor = []
    
    def __init__(self, val, dictionary):
        self.setValue(val)
        self.genChildren(dictionary)
            
    def genChildren(self, dictionary):
        if(isinstance(dictionary, dict)):
            self.successor = dictionary.keys()

    def __str__(self):
        return str(self.value)
    
    def setValue(self, val):
        self.value = val

class DecisionTree:
    def __init__(self):
        self.tree = None
        self.attributes = []
        return;

    # Calculate information gain for given attribute
    # data list is changed to reflect values only for chosen attribute
    def getInformationGain(self, attributes, data, attr, clsResAttr):
        # get index of the attribute
        i = attributes.index(attr)
        # Find frequency of values in attributes
        frequency = {}
        for entry in data:
            if (entry[i] in frequency):
                frequency[entry[i]] += 1.0
            else:
                frequency[entry[i]]  = 1.0

        # Compute sum of the entropy for each subset of data weighted by their probability of occuring in the training set.
        attrSubsetEntropy = 0.0
        for value in frequency.keys():
            valProb = frequency[value] / sum(frequency.values())
            attrSubsetData = []
            for record in data:
                if record[i] == value:
                    attrSubsetData.append(record);
            attrSubsetEntropy += valProb * self.entropy(attributes, attrSubsetData, clsResAttr)

        # Subtract entropy from entropy of entire dataset
        return (self.entropy(attributes, data, clsResAttr) - attrSubsetEntropy)

    # get entropy of the chosen attribute with the given data-set
    def entropy(self, attributes, data, clsResAttr):
        frequency = {}
        dataEntropy = 0.0

        i = attributes.index(clsResAttr)
        # Calculate the frequency of each of the values in given attribute
        for entry in data:
            if (entry[i] in frequency):
                frequency[entry[i]] += 1.0
            else:
                frequency[entry[i]] = 1.0

        # entropy is sum of log of all values for each of the random variable
        import math;
        for freq in frequency.values():
            dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
        return dataEntropy

    # select attibute with high infromation gain
    def selectAttribute(self, data, attributes, cls_index):
        attr_high_ig = attributes[0]
        maxIGain = 0;
        for attr in attributes:
            # cannot select class result as attribute
            if (attr == cls_index):
                continue;
            newIGain = self.getInformationGain(attributes, data, attr, cls_index) 
            if newIGain>maxIGain:
                maxIGain = newIGain
                attr_high_ig = attr
        return attr_high_ig

    # get most frequent value for an attribute
    def get_max_freq_values(self, attributes, data, cls_index):
        #find target attribute
        frequency = {}
        #find target in data
        index = attributes.index(cls_index)
        #calculate frequency of values in target attr
        for record_dt in data:
            if (record_dt[index] in frequency):
                frequency[record_dt[index]] += 1 
            else:
                frequency[record_dt[index]] = 1
        max = 0
        major = ""
        for key in frequency.keys():
            if frequency[key]>max:
                max = frequency[key]
                major = key
        return major

    # get values in the column of the given attribute
    def getAttributeValues(self, data, attributes, attr):
        index = attributes.index(attr)
        values = []
        for entry in data:
            if entry[index] not in values:
                values.append(entry[index])
        return values

    # Get the subset using the selected attribute
    def getChosenSubset(self, data, attributes, attr_high_ig, val):
        examples = []
        index = attributes.index(attr_high_ig)
        for entry in data:
            #find entries with the give value
            if (entry[index] == val):
                newEntry = []
                #add value if it is not in chosen attribute column
                for i in range(0,len(entry)):
                    if(i != index):
                        newEntry.append(entry[i])
                examples.append(newEntry)
        return examples

    def get_positive_rows_count(self, data, attributes, cls_index):
        index = attributes.index(cls_index)
        count_pos = 0        # positive count
        for entry in data:
            #find entries with the give value
            if entry[index] == '1':
                count_pos += 1
        return count_pos

    # build ID3 decision tree from training data
    # The method is called from other method like this: self.buildTree(data, self.attributes, class_res_index, 0)
    def buildTree(self, data, attributes, class_res_index, threshold):
        threshold += 1

        # backup of training data
        data = data[:]

        # was previously: if len(attributes) < 1 or not data:
        # if len(attributes) = 1 then only category/class result is in list of columns
        if len(attributes) <= 1 or not data:
            # This is following line of pseudocode from AIMA, p702
            # if examples is empty then return PLURALITY-VALUE(parent examples)
            # and this line
            # else if attributes is empty then return PLURALITY-VALUE(examples)
            # most common values
            plurality_value = self.get_max_freq_values(attributes, data, class_res_index)
            return plurality_value
        else:
            # attributes values at indices will point to original data array
            # however indices will be pointing to current data array
            # class_res_index will point to class results (what class the tuple belongs to) on original data array
            # so this is basically list of class results at the moment from the tuples we have entering current decision
            cls_index = attributes.index(class_res_index)
            # replacing this line to save space and make more efficient
            # vals = [record[cls_index] for record in data]
            isfirst = True
            ismatch = True
            class_res = ''
            for record in data:
                if isfirst:
                    class_res = record[cls_index]
                    isfirst = False
                else:
                    if class_res != record[cls_index]:
                        ismatch = False
                        break

            # psuedocode from AIMA, p702
            # If all of them contain the same value: the first category/class result
            # else if all examples have the same classification then return the classification

            # if vals.count(vals[0]) == len(vals):
                # return vals[0]
            if ismatch:
                # same class
                return class_res
            else:
                # A?argmaxa ? attributes IMPORTANCE(a, examples)
                # select an attribute based on computation of information gain
                attr_high_ig = self.selectAttribute(data, attributes, class_res_index)
                # Decision tree with found attribute as root
                tree = {attr_high_ig:{}}
    
                S = 0
                N = len(data)
                p = self.get_positive_rows_count(data, attributes, class_res_index)
                n = N - p
                newAttr = attributes[:]
                newAttr.remove(attr_high_ig)
                # create sub-tree, with subnodes for each of the values in the chosen attribute
                # for each value vk of A do
                for val in self.getAttributeValues(data, attributes, attr_high_ig):
                    # exs ?{e : e ?examples and e.A = vk}
                    # Create a subtree for the current value under chosen attribute
                    examples = self.getChosenSubset(data, attributes, attr_high_ig, val)
                    # high gain attribute is removed and returned in examples, attributes should be updated
                    t_k = len(examples)
                    p_k = self.get_positive_rows_count(examples, newAttr, class_res_index)
                    n_k = t_k - p_k
                    p_hat = (p * t_k) / N;
                    n_hat = (n * t_k) / N;
                    pos_value = ((p_hat - p_k) * (p_hat - p_k)) / p_hat
                    neg_value = ((n_hat - n_k) * (n_hat - n_k)) / n_hat
                    S += pos_value + neg_value
                # chi-square test
                if S < chi_square_threshold:
                    # dis not pass chi-squared test
                    # for val in self.getAttributeValues(data, attributes, attr_high_ig):
                        # tree[attr_high_ig][val] = '0'
                    tree[attr_high_ig] = 'nc'
                else:
                    # create sub-tree, with subnodes for each of the values in the chosen attribute
                    # for each value vk of A do
                    for val in self.getAttributeValues(data, attributes, attr_high_ig):
                        # exs ?{e : e ?examples and e.A = vk}
                        # Create a subtree for the current value under chosen attribute
                        examples = self.getChosenSubset(data, attributes, attr_high_ig, val)
                        # subtree?DECISION-TREE-LEARNING(exs, attributes ?A, examples)
                        subtree = self.buildTree(examples, newAttr, class_res_index, threshold)
    
                        # assign this tree as subtree under the decision tree
                        # add a branch to tree with label (A = vk) and subtree
                        tree[attr_high_ig][val] = subtree
        return tree

    # Read training data and build decision tree
    # Store the decision tree in this class
    # This function runs only once when initializing
    # Please read and only read train_data: 'train.data'
    def learn(self, train_data_attrib, train_data_label):
        # read file for retrieving training data
        with open(train_data_attrib, 'r') as attrib_file_read_handle, open(train_data_label, 'r') as label_file_read_handle:
            train_data_attrib_list = attrib_file_read_handle.read().splitlines() 
            train_data_label_list = label_file_read_handle.read().splitlines()

        # clean data, get classes, convert the data into a 2d list
        data = []
        i = 0
        for line in train_data_attrib_list:
            class_res = train_data_label_list[i]
            i += 1
            res = line.split(' ')
            # add the class attribute at the end
            res.append(class_res)
            data.append(res)
        print("Dataset read complete.")
        num_f_rows = len(data)
        num_f_columns = len(data[0])
        print("Number of features: ", num_f_columns, " and number of columns: ", num_f_rows, sep="")
        # get result class index
        # basically class_res_index is class result index
        class_res_index = len(data[0])
        # get attribute indexes, includes result class on right
        
        for i in range(class_res_index):
            self.attributes.append(i);
        # build the tree and save it inside the class for use in solve method
        self.tree = self.buildTree(data, self.attributes, class_res_index-1, 0)
        print("Decision tree building complete")
        return;

    # Add your code here
    # Use the learned decision tree to predict
    def solve(self, query):
        enable_value_matching = False
        # enable_value_matching = True
        entry = query.split(' ')

        tempID3Tree = self.tree.copy()
        result = ""
        while True:
            # for same class plurality value
            if tempID3Tree=='1' or tempID3Tree=='0':
                return tempID3Tree
            
            root = TreeNode(list(tempID3Tree.keys())[0], tempID3Tree[list(tempID3Tree.keys())[0]])
            tempID3Tree = tempID3Tree[list(tempID3Tree.keys())[0]]
            if tempID3Tree=='1' or tempID3Tree=='0':
                return tempID3Tree
            elif tempID3Tree=='nc':
                # we reached here
                # return class for irrelevant attribute
                return '0'
            elif isinstance(tempID3Tree, dict) == False:
                break
                
            if root.value in self.attributes:
                index = self.attributes.index(root.value)
                value = entry[index]
            else:
                # root value is attribute, should be in attributes
                # should never really come here
                return '0'
            
            if value in tempID3Tree.keys():
                child = TreeNode(value, tempID3Tree[value])
                result = tempID3Tree[value]
                tempID3Tree = tempID3Tree[value]
            else:
                if enable_value_matching == False:
                    return '0'
                # unseen value, no classification
                # better to skip this attribute
                # print("value ", value, " not found in keys list ", tempID3Tree.keys())
                int_value = int(value)
                closest_value = ''
                min_diff = 100000000000
                for key_value in tempID3Tree.keys():
                    int_key_value = int(key_value)
                    if int_key_value > int_value:
                        diff = int_key_value - int_value
                    else:
                        diff = int_value - int_key_value
                        
                    if min_diff>diff:
                        min_dff = diff
                        closest_value = key_value
                        
                if closest_value == '':
                    return '0'
                else:
                    value = closest_value
                    child = TreeNode(value, tempID3Tree[value])
                    result = tempID3Tree[value]
                    tempID3Tree = tempID3Tree[value]
                
        return result;

def get_time_diff_from(start_time):
    time_dt_ms = (time.clock() - start_time)*1000
    time_dt_sec = int(time_dt_ms / 1000)
    time_dt_ms = time_dt_ms % 1000
    time_dt_min = int(time_dt_sec / 60)
    time_dt_sec = time_dt_sec % 60
    res_string = str(time_dt_min) + "m " + str(time_dt_sec) + "s"
    return res_string

def DT_solve():
    start_time = time.clock()
    solver = DecisionTree();
    solver.learn('trainfeat.csv', 'trainlabs.csv');
    correct = 0.0;
    total = 0.0;
    print("Time taken to build the tree: ", get_time_diff_from(start_time))
    start_time = time.clock()
    # import cPickle as pickle
    # with open('tree_dict.bin', 'wb') as fp:
      #  pickle.dump(data, fp)
    
    with open("testfeat.csv", "r") as f_data, open("testlabs.csv", "r") as f_label:
        data = f_data.read().splitlines()
        label_list = f_label.read().splitlines()
    i = 0
    for row in data:
        predict = solver.solve(row)
        # print("prediction for ", i, " is ", predict)
        label = label_list[i]
        i += 1
        if label == predict:
            correct = correct + 1
        total = total + 1
    print("Solve time: ", get_time_diff_from(start_time))
    print("Decision Tree accuracy for test against ", i, " data ", ": %f" % (correct / total), sep="")

# entry method 
def main(argv):
    if len(argv) == 3:
        if argv[1] == '-t' or argv[1] == '-threshold':
            global chi_square_threshold
            # threshold 0.01 - critical value 6.635
            if argv[2] == '0.005':
                chi_square_threshold = 7.879
            elif argv[2] == '0.01':
                chi_square_threshold = 6.635
            # threshold 0.05 - critical value 3.841
            elif argv[2] == '0.05':
                chi_square_threshold = 3.841
            # threshold 0.005 - critical value 7.879
            elif argv[2] == '1' or argv[2] == '1.00' :
                chi_square_threshold = 0
        print("Critical value for chi-sqaured criterion: ", chi_square_threshold);
    else:
        print("Wrong command line arguments, using default");

    DT_solve()

if __name__ == "__main__":
    import sys
    main(sys.argv);
