__author__ = 'weiliangxing'

import math
import matplotlib.pyplot as plt

# class Mail:
#     def __init__(self, name, is_spam):
#         self.name = name
#         self.is_spam = is_spam
#         self.feature_list = []
#
#     def get_features(self, feature_list):
#         self.feature_list = feature_list
#
#
# class Feature:
#     def __init__(self, name, freq):
#         self.name = name
#         self.freq = freq
#         # self.mail_list = []
#
#     # def get_mails(self, mail_list):
#     #     self.mail_list = mail_list


class ReadFile:
    """
    Read input file.
    files could be train.txt or test.txt or any other txt files with same format
    """
    def __init__(self, input_file):
        self.input_file = input_file
        with open(self.input_file, 'r') as f:
            read_data = f.read()
        read_data_lines = read_data.splitlines()
        # generate input as list
        self.input_list = self.read_lines(read_data_lines)

    def read_lines(self, lines):
        """
        read splitted lines of input file
        :param lines: splitted lines list of input file
        :return:a list with space split
        """
        input_list = []

        for line in lines:
            single_mail = line.split(" ")
            input_list.append(single_mail)
        return input_list


class PlotResult:
    """
    plot result of accuracy for all three types
    """
    def __init__(self, input_file):
        input_list = ReadFile(input_file).input_list
        coord_x = []
        coord_y_total = []
        coord_y_spam = []
        coord_y_ham = []

        for line in input_list:
            coord_x.append(line[0])
            coord_y_total.append(line[1])
            coord_y_spam.append(line[2])
            coord_y_ham.append(line[3])
        plt.plot(coord_x, coord_y_total)
        plt.plot(coord_x, coord_y_spam)
        plt.plot(coord_x, coord_y_ham)
        plt.legend(['Total accuracy', 'Spam accuracy', 'Ham accuracy'], loc='lower right')
        axes = plt.gca()
        axes.set_xlim([0, 10000])
        axes.set_ylim([0.7, 1.0])
        plt.show()
        print(input_list)


class GenerateTest:
    """
    use classifier for test file and generate accuracy file for each component
    """
    def __init__(self, input_file, classifier):
        self.test_file = input_file
        self.alpha = classifier.alpha
        self.prob_spam = classifier.prob_spam
        self.prob_ham = classifier.prob_ham
        # self.cond_prob is for test the single alpha input
        # self.cond_prob = classifier.cond_prob
        # list of (alphaValue, {feature:(spamNumer,spamDenom),(hamNumer,hamDenom)})
        self.cond_prob_list = classifier.cond_prob_list
        # for prob in self.cond_prob_list:
        #     print(prob)

        self.input_list = ReadFile(self.test_file).input_list

        self.test_result_list = []
        self.accuracy_list = []
        self.classify()

    def write_output(self):
        """
        write the results of accuracy of classifier to file
        note: here for experiments, we keep append new results to existing results
        format in each line means: smooth parameter, accuracy for all mails, accuracy for spam, accuracy for ham.
        :return: None
        """
        with open("result.txt", "a") as f:
            for item in self.accuracy_list:
                alpha = item[0]
                total_accuracy = item[1]
                spam_accuracy = item[2]
                ham_accuracy = item[3]
                f.write("{var1} {var2} {var3} {var4}\n".
                        format(var1=alpha, var2=total_accuracy, var3=spam_accuracy, var4=ham_accuracy))
            f.close


    def classify(self):
        """
        the concrete function to do classfication for test file
        :return: refresh self.accuracy_list which is most important for analysis
        """
        for prob in self.cond_prob_list:
            single_alpha = prob[0]
            cond_prob_dic = prob[1]
            test_result = []
            total_num = len(self.input_list)
            match_num = 0
            total_spam = 0
            total_ham = 0
            match_spam = 0
            match_ham = 0
            for mail in self.input_list:
                case_name = mail[0]
                case_type = mail[1]
                if case_type == "spam":
                    total_spam += 1
                else:
                    total_ham += 1
                # use log calculation to avoid underflow
                spam_log_prob = self.generate_log_prob("spam", mail[2:], self.prob_spam, cond_prob_dic)
                ham_log_prob = self.generate_log_prob("ham", mail[2:], self.prob_ham, cond_prob_dic)
                if spam_log_prob <= ham_log_prob:
                    learn_type = "ham"
                else:
                    learn_type = "spam"

                if learn_type == case_type:
                    if case_type == "spam":
                        match_spam += 1
                    if case_type == "ham":
                        match_ham += 1
                    match_num += 1

                test_result.append((case_name, case_type, spam_log_prob, ham_log_prob, learn_type))
            self.test_result_list.append((single_alpha, test_result))
            accuracy = (single_alpha, float(match_num / total_num), float(match_spam / total_spam), float(match_ham / total_ham))
            self.accuracy_list.append(accuracy)

        # print("======")
        # print(self.test_result_list)
        # print(self.accuracy_list)

    def generate_log_prob(self, mail_type, features, prior, cond_prob_dic):
        """
        function to calculate the log result for each test mail with different type
        :param mail_type: "spam" or "ham"
        :param features: feature list for each mail; same format as test.txt
        :param prior: prior for spam or ham probability
        :param cond_prob_dic: dictionary for each feature which has (spam condition prob, ham condition prob)
        :return: log probability for spam or ham
        """
        log_result = 0.0
        log_result += math.log(prior)

        if mail_type == "spam":
            for i in range(1, len(features), 2):
                single_feature = features[i-1]
                freq = float(features[i])
                if single_feature in cond_prob_dic:
                    cond_prob = cond_prob_dic[single_feature][0][0] / cond_prob_dic[single_feature][0][1]
                    log_result += freq * math.log(cond_prob)
                else:
                    log_result += 0
        else:
            for i in range(1, len(features), 2):
                single_feature = features[i-1]
                freq = float(features[i])
                if single_feature in cond_prob_dic:
                    cond_prob = cond_prob_dic[single_feature][1][0] / cond_prob_dic[single_feature][1][1]
                    log_result += freq * math.log(cond_prob)
                else:
                    log_result += 0

        return log_result



class GenerateClassifier:
    """
    class to generate naive bayes classifier
    """
    def __init__(self, input_file, alpha):

        self.feature_dic_list = [{}, {}, {}]
        self.selected_features_dic = {}
        self.alpha = alpha

        self.input_file = input_file

        # generate input as list
        self.input_list = ReadFile(self.input_file).input_list
        # generate feature dic with all dic, dic for spam and dic for ham
        self.feature_dic_list = self.generate_feature_dic(self.input_list)
        # preprocess1: eliminate those whose number is less than three
        self.feature_dic_list_preprocess_1 = self.eliminate_feature(self.feature_dic_list)
        # result shows the original data set already eliminated

        # preprocess2: select top 500 features with largest MI information
        self.feature_count_dic = self.generate_feature_count_dic(self.input_list)
        # generate prob_spam, prob_ham, condition prob for each 500 features for spam and ham.
        output_tuple = self.mutual_information(self.input_list, self.feature_dic_list_preprocess_1)
        self.prob_spam = output_tuple[0]
        self.prob_ham = output_tuple[1]
        if type(self.alpha) is list:
            # self.cond_prob will be a list with several output dictionary
            self.cond_prob_list = []
            for a in self.alpha:
                cond_prob_dic = self.cond_prob(output_tuple, self.feature_dic_list_preprocess_1, a)
                self.cond_prob_list.append((a, cond_prob_dic))

        else:
            # self.cond_prob will be a dictionary
            # self.cond_prob = {}  why add this line will throw error???
            self.cond_prob = self.cond_prob(output_tuple, self.feature_dic_list_preprocess_1, self.alpha)

    def generate_feature_count_dic(self, input_list):
        """
        function to generate counting for each feature.
        Note: here counting means the number of presents for each feature, not the frequency
        :param input_list: input list of input file
        :return: a dictionary with feature as key, (#presents in spam, #presents in ham) as value
        """
        dic = {}
        for m in input_list:
            mail_type = m[1]
            for i in range(3, len(m), 2):
                if mail_type == "spam":
                    if m[i-1] in dic:
                        # left for spam counting, right for ham counting
                        dic[m[i-1]] = (dic[m[i-1]][0] + 1, dic[m[i-1]][1])
                    else:
                        dic[m[i-1]] = (1, 0)
                else:
                    if m[i-1] in dic:
                        # left for spam counting, right for ham counting
                        dic[m[i-1]] = (dic[m[i-1]][0], dic[m[i-1]][1] + 1)
                    else:
                        dic[m[i-1]] = (0, 1)
        return dic

    def mutual_information(self, input_list, feature_num_list):
        """
        function to calculate mutual information for feature selection
        :param input_list: input list for input file
        :param feature_num_list: list with [total_dic, spam_dic, ham_dic] for each feature as key
        :return:prob_spam, prob_ham, count_spam, count_ham, final_feature_dic with meaning as it is named
        """
        spam_feature_list = []
        final_feature_dic = {}

        N_spam = 0.0
        N = len(input_list)
        for m in input_list:
            if m[1] == "spam":
                N_spam += 1.0
        N_ham = N - N_spam
        dic = self.feature_count_dic

        for f in dic:
            mi_spam = 0.0
            if dic[f][0] != 0:
                mi_spam += dic[f][0] * math.log((N * dic[f][0])/((dic[f][0] + dic[f][1]) * N_spam))
            if (N_spam - dic[f][0]) != 0:
                mi_spam += (N_spam - dic[f][0]) * math.log((N * (N_spam - dic[f][0]))/((N - (dic[f][0] + dic[f][1])) * N_spam))
            if dic[f][1] != 0:
                mi_spam += dic[f][1] * math.log((N * dic[f][1]) / ((dic[f][0] + dic[f][1]) * N_ham))
            if (N_ham - dic[f][1]) != 0:
                mi_spam += (N_ham - dic[f][1]) * math.log((N * (N_ham - dic[f][1]))/((N - (dic[f][0] + dic[f][1])) * N_ham))
            mi_spam /= N
            spam_feature_list.append([mi_spam, f])

        def get_key(item):
            return item[0]

        selected_features = sorted(spam_feature_list, key=get_key, reverse=True)
        selected_features = selected_features[:500]

        for f in selected_features:
            # left is the presentation times in spam, not number; right is same;
            final_feature_dic[f[1]] = (dic[f[1]][0], dic[f[1]][1])

        # print(final_feature_dic)

        # build conditional probabilities
        prob_spam = N_spam / N
        prob_ham = N_ham / N

        # print(prob_spam)
        # print(prob_ham)

        [total_dic, spam_dic, ham_dic] = feature_num_list
        count_spam = 0
        count_ham = 0
        count_total = 0
        for f in spam_dic:
            count_spam += spam_dic[f]
        for f in ham_dic:
            count_ham += ham_dic[f]
        for f in total_dic:
            count_total += total_dic[f]

        # print(count_spam)
        # print(count_ham)
        # print(count_total)

        return prob_spam, prob_ham, count_spam, count_ham, final_feature_dic

    def cond_prob(self, input_tuple, feature_num_list, alpha):
        """
        function to calculate the conditional probability for each feature
        :param input_tuple: return value of function mutual_information
        :param feature_num_list: dic list [total_dic, spam_dic, ham_dic] with features as key
        :param alpha: smooth parameter. It can be float value or a list of float value.
        :return: dictionary with feature as key, ((numer_1 of spam, denom_1 of spam),
                (numer_2 of ham, denom_2 of ham)) as value. They are conditional probabilities.
        """
        [total_dic, spam_dic, ham_dic] = feature_num_list
        count_spam = input_tuple[2]
        count_ham = input_tuple[3]
        final_feature_dic = input_tuple[4]
        num_features = 500
        output_dic = {}
        for f in final_feature_dic:
            if f in spam_dic:
                numer_1 = spam_dic[f] + alpha
            else:
                numer_1 = 0 + alpha
            denom_1 = count_spam + alpha * num_features
            if f in ham_dic:
                numer_2 = ham_dic[f] + alpha
            else:
                numer_2 = 0 + alpha
            denom_2 = count_ham + alpha * num_features
            output_dic[f] = ((numer_1, denom_1), (numer_2, denom_2))
        # print(output_dic)
        return output_dic

    def generate_feature_dic(self, input_list):
        """
        generate dictioanry for each feature
        Note: here value in each dictionary's value is total frequency of each feature
        :param input_list: input list of input file
        :return:[dic, spam_dic, ham_dic] with feature as value
        """
        dic = {}
        spam_dic = {}
        ham_dic = {}
        for m in input_list:
            mail_type = m[1]
            for i in range(3, len(m), 2):
                if mail_type == "spam":
                    if m[i-1] in spam_dic:
                        # spam_dic[m[i-1]] = (spam_dic[m[i-1]][0] + int(m[i]), "spam")
                        spam_dic[m[i-1]] += int(m[i])
                    else:
                        spam_dic[m[i-1]] = int(m[i])
                else:
                    if m[i-1] in ham_dic:
                        ham_dic[m[i-1]] += int(m[i])
                    else:
                        ham_dic[m[i-1]] = int(m[i])

                if m[i-1] in dic:
                    dic[m[i-1]] += int(m[i])
                else:
                    dic[m[i-1]] = int(m[i])

        return [dic, spam_dic, ham_dic]

    def eliminate_feature(self, dic_list):
        """
        fucntion to eliminate the feature who has less 3 time in total frequency
        :param dic_list: the dictionary with feature as key and total frequency as value
        :return: [dic, spam_dic, ham_dic] with feature as key
        """
        [dic, spam_dic, ham_dic] = dic_list
        for key in dic:
            if dic[key] < 3:
                if key in spam_dic:
                    del spam_dic[key]
                if key in ham_dic:
                    del ham_dic[key]
                del dic[key]
            else:
                continue
        return [dic, spam_dic, ham_dic]


def main():

    # classifier = GenerateClassifier("data/train.txt", [0.5, 1.0])
    # classifier = GenerateClassifier("data/train.txt", 1.0)
    # classifier = GenerateClassifier("data/train.txt", [10000.0])

    # comment out each range when finishing experiment.
    # range_0 = [x*0.1 for x in range(1, 10)]
    # range_1 = [x*10 for x in range(1, 11)]
    # range_1.insert(0, 1)
    # range_2 = [x*100 for x in range(2, 11)]
    # range_3 = [x*1000 for x in range(2, 11)]
    # range_4 = [x*10000 for x in range(2, 11)]
    range_5 = [x*100000 for x in range(2, 11)]

    try_range = range_5
    classifier = GenerateClassifier("data/train.txt", try_range)
    result = GenerateTest("data/test.txt", classifier)
    result.write_output()
    print("Done.")

    draw_result = PlotResult("result.txt")

    # print(result.accuracy_list)
    # print(train_model.feature_count_dic)
    # print(train_model.feature_dic_list[0])
    # print(len(train_model.feature_count_dic))
    # print(len(train_model.feature_dic_list[0]))
    # print(len(train_model.feature_dic_list[1]))
    # print(len(train_model.feature_dic_list[2]))
    # print(len(train_model.feature_dic_list_preprocess_1[0]))
    # print(len(train_model.feature_dic_list_preprocess_1[1]))
    # print(len(train_model.feature_dic_list_preprocess_1[2]))
    # print(train_model.feature_dic["need"])
    # print(train_model.feature_dic["over"])
    # print(train_model.mails[0].feature_list[0].freq)
    # print(len(train_model.mails[0].feature_list))

if __name__ == "__main__":
    main()