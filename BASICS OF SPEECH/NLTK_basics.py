# Noise Removal : Any data that is not relevant with respect to the main topic is considered as noise.

noise_list = ["I","am"]
def remove_noise(input_text):
   words = input_text.split()
   noise_free_words = [word for word in words if word not in noise_list]
   noise_free_text = " ".join(noise_free_words)
   return noise_free_text


sample_text = "I am Ganapathi Subramanyam"
cleaned_output = remove_noise(sample_text)
print(cleaned_output)

#Stemming and Lemmetization :  It is a process through which you can revert a word to its root form.
import nltk
from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()
word = "helping" 
stem.stem(word)

#Object Standarization : A text data also contains some phrases short forms that are not in lexical dictionary and these keywords are not understand by search engines.
lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love"}
def _lookup_words(input_text):
   words = input_text.split()
   new_words = []
   for word in words:
     if word.lower() in lookup_dict:
       word = lookup_dict[word.lower()]
     new_words.append(word) 
     new_text = " ".join(new_words)
     print(new_text)
     #return new_text
_lookup_words("luv this is a dm tweet by Ganapathi")

# Parts of Speech Tagging : This will seperate the word and assign parts of speech accordingly.
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, pos_tag
text = "I am learning Natural Language Processing"
tokens = word_tokenize(text)
print(pos_tag(tokens))

#predicting the comment is positive or negative
from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob import TextBlob
training_corpus = [
                    ('I am exhausted of this work.', 'Class_B'),
                    ("I dont like icecream", 'Class_B'),
                    ('He is my worst enemy!', 'Class_B'),
                    ('My management is poor.', 'Class_B'),
                    ('I love this pizza.', 'Class_A'),
                    ('This is an brilliant place!', 'Class_A'),
                    ('I feel very good about these dates.', 'Class_A'),
                    ('This is my best work.', 'Class_A'),
                    ("What an awesome view", 'Class_A'),
                    ('I do not like this dish', 'Class_B')]
test_corpus = [
                 ("I am not feeling well today.", 'Class_B'), 
                 ("I feel brilliant!", 'Class_A'), 
                 ('Gary is a friend of mine.', 'Class_A'), 
                 ("I can't believe I'm doing this.", 'Class_B'),
                 ('The date was good.', 'Class_A'), ('I do not enjoy my job', 'Class_B')]
model = NBC(training_corpus) 
print(model.classify("Their codes are worst."))
print(model.classify("i love Python"))

#ngrams: splitting a sentence into words
def generate_ngrams(text, n):
     words = text.split()
     output = []  
     for i in range(len(words)-n+1):
         output.append(words[i:i+n])
     return output

generate_ngrams('i will be going to Coimbatore', 3)

