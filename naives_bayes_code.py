#importing libraries
import re #regex library
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

print('hello')
#Prepocessing the data
def preprocess_data(str_data)->str:
    '''
    Parameters: 

    str_data : input string to be preprocessed 

    This function will do the following things on the input string.
    1. It will remove all characters except letters and spaces.
    2. It will remove multispaces and replace it by single space.
    3. It will change the data into lowercase


    Returns:

    preprocessed string
    '''

    cleaned_str:str = ''
    cleaned_str = re.sub('[^a-z/s]+', ' ', str_data, flags=re.IGNORECASE)
    cleaned_str = re.sub('(/s+)', ' ', cleaned_str)
    cleaned_str = cleaned_str.lower()

    return cleaned_str


# class for Naives bayes that contains the methods related to this algorithm
class Naives_bayes:

    #inintialising the cinstructor wuth passing the uniques classes only
    def __init__(self, unique_classes):
        self.classes = unique_classes #classes

    
    def addToBoW(self, example, dict_index)->None:

        '''
        Parameters:
        1. example -> sentence/example in the dataset
        2. dict_index -> Index value of class where the example/sentence belongs to

        Function's work:
        This function will split the example into words(tokens) as space as separator and add the words count in the
        respective class/label dictionary.

        Returns: 
        nothing
        '''

        if isinstance(example, np.ndarray): example = example[0]

        for word in example.split():
            self.bow_dicts[dict_index][word] +=1 #increasing the count of word by 1


    def train(self, dataset, labels):

        '''
        Parameters:
        1. dataset -> The dataset for training the Naives Bayes model.
        2. labels -> Labels for classifying purpose.

        Function's work:
        This function will create BoW for each category and make it use for training the model

        Returns:
        nothing
        '''

        self.examples = dataset
        self.labels= labels
        self.bow_dicts = np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])]) #creating BoW for each classes

        if not isinstance(self.examples, np.ndarray):
            self.examples = np.array(self.examples)

        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)

        for cat_index, cat in enumerate(self.classes):

            all_cat_examples = self.examples[self.labels == cat]

            cleaned_examples = [preprocess_data(cat_examples) for cat_examples in all_cat_examples]
            cleaned_examples = pd.DataFrame(data=cleaned_examples)

            np.apply_along_axis(self.addToBoW, 1, cleaned_examples, cat_index)


        '''
        Next, we are going to calculate the probability values needed for the Niave bayes algorithm.
        We are going to find the following.
        1. p(c) -> Probabilities of each class.
        2. count of words for each class.
        3. Vocabulary |v|


        '''

        prob_classes = np.empty(self.classes.shape[0])
        cat_words_count = np.empty(self.classes.shape[0])

        all_words = []

        for cat_index, cat in enumerate(self.classes):

            #probabilities for each class
            prob_classes[cat_index] = np.sum(self.labels == cat) / float(self.labels.shape[0])

            #count of words for each class
            cat_words_count[cat_index] = np.sum(np.array(list(self.bow_dicts[cat_index].values()))) + 1

            #get all words of the category
            all_words += list(self.bow_dicts[cat_index].keys())

        
        #Making the vocabulary by get unique words in all words
        self.vocab = np.unique(np.array(all_words)) #V
        self.vocab_length = self.vocab.shape[0] # |V|

        #Findind the numerator value: count of each class + |V| + 1
        denom = np.array([cat_words_count[cat_index] + self.vocab_length + 1 for cat_index in range(self.classes.shape[0])])

        '''
        We have find out the probabilities and classes info. We are make these all info in a single array.
        '''

        self.cat_info = [(self.bow_dicts[cat_index], prob_classes[cat_index], denom[cat_index]) for cat_index, cat in enumerate(self.classes)]
        self.cat_info = np.array(self.cat_info)


    
    def getTestProb(self, test_example):

        '''
        Parameters:
        1. test_example -> Each example from the test data set(preprocessed).

        What will the function do?
        This function will find the posterior probability of the test example by probability values.

        Returns:
        posterior probability for example for each class as narray

        
        '''

        #Calculating the likelihood probability for each classes
        likelihood_prob = np.zeros(self.classes.shape[0]) # to store the likelihood probabilities for each classes

        for cat_index, cat in enumerate(self.classes):
            for token in test_example.split():
                #getting the count of each word and adding plus for the numerator
                test_token_count = self.cat_info[cat_index][0].get(token, 0) + 1

                #calculating the likelihood probabilty for the word
                test_token_prob = test_token_count / float(self.cat_info[cat_index][2])

                #We know that to avoid underflow error, we are taking log for the likelihood probability.
                likelihood_prob[cat_index] += np.log(test_token_prob)


        #calculating posterior probability
        post_prob = np.empty(self.classes.shape[0]) # to store the posterior probabilities for each classes

        for cat_index, cat in enumerate(self.classes):
            #calculating the posterior probablity by adding with the log of likelihood probabilty
            post_prob[cat_index] = np.log(self.cat_info[cat_index][1]) + likelihood_prob[cat_index]

        return post_prob
    
    def test(self, test_set):

        '''
        Parameters:
        1. test_set -> This is the test set given for classifying them into t=one of the classes.

        What will the function do ? 
        This funcion will calculate the posterior probablity for each examples and classify them as predictions

        Returns:
        return the predictions list
        '''


        predictions = [] #to store the each predictions of each example
        for example in test_set:

            #preprocessing the example
            cleaned_example = preprocess_data(example)

            #finding the post probability for the example
            post_prob = self.getTestProb(cleaned_example)

            #Prediction the class of the example by class which having the maximum posterior probability
            predictions.append(self.classes[np.argmax(post_prob)])

        return predictions


''''
We are going to load a dataset from sklean library.
Dataset :
    It's a newsgroup dataset which contains the newsgroups post on 20 topics.
    So, the dataset will have 20 classes.
    But, we will be only taking four classes of them for training and testing.
'''




# #training the dataset
# categories=['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med'] 
# newsgroup_train = fetch_20newsgroups(subset = 'train', categories=categories)

# train_data = newsgroup_train.data
# train_label = newsgroup_train.target

# #iniating a object of Naive Bayes class
# nb = Naives_bayes(np.unique(train_label))

# #training the newsgroup data
# print("---------Training started-----------")
# nb.train(train_data, train_label)

# print("---------Training completed----------")


# #loading the dataset for testing
# newsgroup_test = fetch_20newsgroups(subset='test', categories=categories)

# test_data = newsgroup_test.data
# test_labels = newsgroup_test.target

# print(f'Number of test examples: {len(test_data)}')
# print(f'Number of test labels: {len(test_labels)}')


# #testing the accuracy of model by validating with the predictions made and the test data

# #Calculating the probabilities and preedicting the data
# pred_classes = nb.test(test_data)


# #calculating the efficiency by comparing with the test data
# efficiency = np.sum(pred_classes == test_labels)/float(test_labels.shape[0]) * 100

# print(f"Efficiency of the NB model: {efficiency} %")



'''
Checking the models performance with Kaggle dataset

'''
#Loading the Kaggle train dataset
training_data = pd.read_csv("labeledTrainData.tsv", sep = '\t')

#We separated the review and sentiment data
x_train = training_data['review'].values
y_train = training_data['sentiment'].values


train_data, test_data, train_labels, test_labels = train_test_split(x_train, y_train, test_size=0.25, shuffle=True, random_state=42, stratify=y_train)

classes = np.unique(train_labels)

#iniating another object for Naives bayes model
nb1 = Naives_bayes(classes)

print("---------Training started-----------")
nb1.train(train_data, train_labels)
print("---------Training completed---------")

#Predicting the classes for the test data
predicted_classes = nb1.test(test_data)

#checking for the efficiency of the Naives Model on the Kaggle dataset
efficiency = np.sum(predicted_classes == test_labels)/float(test_labels.shape[0]) * 100

print(f"The efficiency of the model on Kaggle dataset: {efficiency} %")





