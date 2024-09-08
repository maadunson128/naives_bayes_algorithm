#importing libraries
import re #regex library
import numpy as np
import pandas as pd
from collections import defaultdict


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



class Naives_bayes:

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
        self.bow_dicts = np.array([defaultdict(lambda:0) for index in (self.classes.shape(0))]) #creating BoW for each classes

        if not isinstance(self.examples, np.ndarray):
            self.examples = np.array(self.examples)

        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)

        for cat_index, cat in enumerate(self.classes):

            all_cat_examples = self.examples[self.classes == cat]

            cleaned_examples = [preprocess_data(cat_examples) for cat_examples in all_cat_examples]
            cleaned_examples = pd.DataFrame(data=cleaned_examples)

            np.apply_along_axis(self.addToBoW, 1, cleaned_examples, cat_index)




