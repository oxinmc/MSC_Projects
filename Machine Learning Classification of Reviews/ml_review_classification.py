################################################################################################################################
# Section 1
# Import Libraries

import urllib.request           #For opening links to reviews
import pandas as pd             #For reading data from .txt files
import seaborn as sns           #For creating heatmap of confusion matrix after classification
import matplotlib.pylab as plt  #For plotting graphs e.g. confusion matrix heatmap
import numpy as np              #For creating list to test effects of test size

#Tools related to the classification process
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score


################################################################################################################################
# Section 2
# Define necessary functions

#Function for finding and saving a list of URLs for all reviews corresponding to each company within a chosen category
def url_collection(link):
    
    urls = []

    response = urllib.request.urlopen(link) #Open link
    html = response.read().decode()

    if response.code == 200: #Request succeeded, and the resource is returned.
        
        lines = html.strip().split("\n")

        for l in lines:       

            if l.startswith('\t\t\t<p><h5>&mdash;') == True: #All company review websites start with this
                
                head, sep, tail = l.partition('\'>') #Seperate URL text from jargon
                head, sep, tail = head.partition('=\'')

                urls.append(tail)


    elif response.code == 404:
        print('The requested resource does not exist.')

    elif response.code == 500:
        print('An unexpected error happened on the server side.')

    else:
        print('The resource has moved to another URL.') #For codes 301/302/303

    return(urls)

#####################################################

#Function for collecting all review data
def collect_data(urls):
    
    all_reviews = 'Text\tAttitude' #Create titles for stored review text
    star_info = []
    
    for i in urls: #Run through all companies within each category for review scraping

        link = 'http://mlg.ucd.ie/modules/yalp/{url}'.format(url=i)
        response = urllib.request.urlopen(link) #Open URL of company's reviews
        html = response.read().decode()


        lines = html.strip().split("\n")

        for l in lines:

            if l.startswith('\t\t\t<p class=\'review-start\'') == True: #What all review lines start with

                    head, sep, tail = l.partition('-star\'><')
                    head, sep, tail = head.partition('alt=\'')
                    stars = tail #Find the star value incase this is of interest later on 

                    if int(tail) > 3:
                        attitude = '1' #Positive Review
                    else:
                        attitude = '0' #Negative Review

                    head2, sep2, tail2 = l.partition('class=\'review-text\'>')
                    head2, sep2, tail2 = tail2.partition('</p>') #Text seperated from extra, unnecessary code
                    
                    #To fix code's interpretation of characters, e.g.
                    # ' as &#x27; // " as &quot; // & as &amp; // and for pandas interpretation of the $ sign

                    text = head2.replace('&#x27;','\'').replace('&quot;','"').replace('&amp;','&').replace('$', '\$')

                    star_info.append(stars)
                    all_reviews = all_reviews + '\n' + str(text) + '\t' + str(attitude) #Structure data for easy use later on
                    
    return(all_reviews, star_info)

#####################################################

#Function for saving info as .txt files
def save_file(name, data):
    with open(name, "w", encoding="utf-8") as output:
        for i in data:
            output.write(i) #Fill file with data

#####################################################

#Function for preprocessing review data from file and applying classification via the Naive Bayes classifier
#This classification's performance is then evaluated
def naive_bayes_classify(file): #file:"cafe_reviews.txt"
    
    model = MultinomialNB() #Prepare the Naive Bayes classification model
    
    data = pd.read_csv(file, sep="\t") #Read in data file
    reviews = data["Text"] #Seperate dat into reviews and target classification
    target = data["Attitude"]
    
    vectorizer = TfidfVectorizer(stop_words="english",min_df = 10) #min_df: filter terms appearing in less than 'x' reviews
    X = vectorizer.fit_transform(reviews)

    #Train for learning/Test for label prediction
    data_train, data_test, target_train, target_test = train_test_split(X, target, test_size=0.3)
    
    
    model.fit(data_train, target_train)  #Building the classification model
    predicted = model.predict(data_test) #Classification's 'best-guess' of the test set labels
    
    cm = confusion_matrix(target_test, predicted, labels=[0,1]) #For representing false positives etc.
    evaluation = classification_report(target_test, predicted, target_names=["negative","positive"])# output_dict=True
    acc_scores = cross_val_score(model, X, target, cv=4, scoring="accuracy") #k-fold cross validation
    
    acc = acc_scores.mean() #Mean accuracy of model's predictions
    std = acc_scores.std()  #Standard deviation of model's predictions

    return(cm, evaluation, acc, std) #Returns confusion matrix for later plotting 

#####################################################

#Function for plotting a heatmap of a 2x2 confusion matrix
def plot_cm(cm, plot_title):
    
    plt.figure()
    ax = sns.heatmap(cm, linewidth=0, vmin=0, vmax=600, annot=True, cmap="Blues", fmt="d")
    ax.xaxis.set_ticklabels(['negative', 'positive']); ax.yaxis.set_ticklabels(['negative', 'positive'])
    plt.title(plot_title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
#####################################################

#Function for displaying the effect of min_df on the accuracy of the classification model
def effect_of_min_df(file, min_df, plot_title):
    
    model = MultinomialNB()
    
    data = pd.read_csv(file, sep="\t")
    reviews = data["Text"]
    target = data["Attitude"]
    acc = []
    
    for i in min_df:
        
        a = 0
        
        for j in range(0,10): #Looped 10 times for every min_df value for averaging and producing a better accuracy
        
            vectorizer = TfidfVectorizer(stop_words="english", min_df = i)
            X = vectorizer.fit_transform(reviews)

            data_train, data_test, target_train, target_test = train_test_split(X, target, test_size=0.3)

            model.fit(data_train, target_train)
            predicted = model.predict(data_test)
            a = a + accuracy_score(target_test, predicted)
            
        b = a/10 #Average
        acc.append(b) #Calculate and append accuracy list
    
    plt.figure()
    plt.plot(min_df, acc)
    plt.title(plot_title)
    plt.xlabel('Document Frequency')
    plt.ylabel('Accuracy of Classification')
    plt.show()

#####################################################

#Function for displaying the effect of the test size on the accuracy of the classification model
def effect_of_test_size(file, test_size, plot_title):
    
    model = MultinomialNB()
    
    data = pd.read_csv(file, sep="\t")
    reviews = data["Text"]
    target = data["Attitude"]
    acc = []
    
    vectorizer = TfidfVectorizer(stop_words="english", min_df = 10)
    X = vectorizer.fit_transform(reviews)
    
    for i in test_size:
        
        a = 0
        
        for j in range(0,100): #Looped 100 times for every test_size value for averaging and producing a better accuracy
            
            data_train, data_test, target_train, target_test = train_test_split(X, target, test_size=i)

            model.fit(data_train, target_train)
            predicted = model.predict(data_test)
            a = a + accuracy_score(target_test, predicted)
        
        b = a/100 #Average
        acc.append(b) #Calculate and append accuracy list
    
    plt.figure()
    plt.plot(test_size, acc)
    plt.title(plot_title)
    plt.xlabel('Test Size')
    plt.ylabel('Accuracy of Classification')
    plt.show()

#####################################################

#Function to evaluate the performance of a classification model on two other categories
def evaluate_a_on_b_and_c(file_a, file_b, file_c):
    
    model = MultinomialNB()
    vectorizer = TfidfVectorizer(stop_words="english", min_df = 10)
    
    #Data is read in from all three categories and preprocessed
    
    ## a
    data_a = pd.read_csv(file_a, sep="\t")
    reviews_a = data_a["Text"]
    target_a = data_a["Attitude"]
    
    X_a = vectorizer.fit_transform(reviews_a)
    #All data is used to train classifier
    data_train_a, data_test_a, target_train_a, target_test_a = train_test_split(X_a, target_a, test_size=0.0)
    
    ## b
    data_b = pd.read_csv(file_b, sep="\t")
    reviews_b = data_b["Text"]
    target_b = data_b["Attitude"]
    
    X_b = vectorizer.transform(reviews_b)
    #Essentially all data is used for testing
    data_train_b, data_test_b, target_train_b, target_test_b = train_test_split(X_b, target_b, test_size=0.999)     
    
    ## c
    data_c = pd.read_csv(file_c, sep="\t")
    reviews_c = data_c["Text"]
    target_c = data_c["Attitude"]
    
    X_c = vectorizer.transform(reviews_c)
    #Essentially all data is used for testing
    data_train_c, data_test_c, target_train_c, target_test_c = train_test_split(X_c, target_c, test_size=0.999)    

    #Classification model is developed on one category, 'a', and tested on category 'b' and 'c'
    
    model.fit(data_train_a, target_train_a)  #Model classifier on data set 'a'
    predicted_b = model.predict(data_test_b) #Test classifcation model on data 'b'
    predicted_c = model.predict(data_test_c) #Test classifcation model on data 'c'
    
    evaluation_b = classification_report(target_test_b, predicted_b, target_names=["negative","positive"])
    evaluation_c = classification_report(target_test_c, predicted_c, target_names=["negative","positive"])
    
    acc_aonb = accuracy_score(target_test_b, predicted_b)
    acc_aonc = accuracy_score(target_test_c, predicted_c)

    #Classification reports are returned as well as accuracy scores
    return(evaluation_b, acc_aonb, evaluation_c, acc_aonc)


################################################################################################################################
# Section 3
'''
Collect all URLs for each company within each category.
Get all review data for cafes, gyms and restaraunts and whether these reviews are positive or negative.
Save these as .txt files.
'''

cafe_urls = url_collection('http://mlg.ucd.ie/modules/yalp/cafes_list.html')
gym_urls = url_collection('http://mlg.ucd.ie/modules/yalp/gym_list.html')
restaraunt_urls = url_collection('http://mlg.ucd.ie/modules/yalp/restaurants_list.html')

cafe_total_review = collect_data(cafe_urls)
gym_total_review = collect_data(gym_urls)
restaraunt_total_review = collect_data(restaraunt_urls)

save_file("cafe_reviews.txt", cafe_total_review[0])
save_file("gym_reviews.txt", gym_total_review[0])
save_file("restaraunt_reviews.txt", restaraunt_total_review[0])


################################################################################################################################
# Section 4
'''
Preprocess the data so it is suitable for classification.
Build a classification model to distinguish between “positive” and “negative” reviews using the Naive Bayes classifier.
Test the predictions of the classification model using an appropriate evaluation strategy.
'''

cafe_classification = naive_bayes_classify("cafe_reviews.txt")
print('--o-- Cafe Reviews: Classification Report --o--\n\n', cafe_classification[1], 
      '\nAccuracy = %.3f' % cafe_classification[2], '\nStandard Deviation = %.3f\n\n' % cafe_classification[3])

gym_classification = naive_bayes_classify("gym_reviews.txt")
print('--o-- Gym Reviews: Classification Report --o--\n\n', gym_classification[1], 
      '\nAccuracy = %.3f' % gym_classification[2],  '\nStandard Deviation = %.3f\n\n' % gym_classification[3])

restaraunt_classification = naive_bayes_classify("restaraunt_reviews.txt")
print('--o-- Restaraunt Reviews: Classification Report --o--\n\n', restaraunt_classification[1], 
      '\nAccuracy = %.3f' % restaraunt_classification[2], '\nStandard Deviation = %.3f\n\n' % restaraunt_classification[3])


################################################################################################################################
# Section 5
'''
Plot of confusion matrix.
'''

plot_cm(cafe_classification[0], 'Cafe Reviews')
plot_cm(gym_classification[0], 'Gym Reviews')
plot_cm(restaraunt_classification[0], 'Restaraunt Reviews')


################################################################################################################################
# Section 6
'''
Investigate effect of minimum document frequency (min_df) and test size (test_size) on the overall accuracy of a classifier.
'''

#Due to the averaging within the 'effect_of_min_df()' function, this section of code may take a while to run
min_df = range(0,50)
effect_of_min_df("cafe_reviews.txt", min_df, 'Cafe Reviews')

test_size = np.linspace(0.01,0.9,16)
effect_of_test_size("cafe_reviews.txt", test_size, 'Cafe Reviews')


################################################################################################################################
# Section 7
'''
Histogram of star rating for each category.

Note: Although 'collections' isn't one of the third-party packages allocated to this assignment, I felt it necessary to use for my own comprehension of the review data being read in and feel it helps me in discussing the quality of the classification models created. If this is not allowed, ignore section 2.4 of this code.
'''

from collections import Counter

star_lists = [cafe_total_review, gym_total_review, restaraunt_total_review]
x = 0

for i in star_lists:
    
    x = x+1
    
    star_list = sorted(i[1])
    counts = Counter(star_list)
    
    labels, values = zip(*Counter(star_list).items())
    indexes = np.arange(len(labels))
    
    if x == 1:
        plot_title = 'Cafe Reviews'
    elif x == 2:
        plot_title = 'Gym Reviews'
    elif x == 3:
        plot_title = 'Restaraunt Reviews'
    
    plt.figure()
    plt.title(plot_title)
    plt.bar(indexes, values)
    #for i, v in enumerate(values):
    #    plt.text(v, i, str(v), color='blue', fontweight='bold')
    for index,data in enumerate(values):
        plt.text(x=index-0.15, y =data+8 , s=f"{data}", fontdict=dict(fontsize=10))
    plt.xticks(indexes, labels) #Label x-axis 1-5
    plt.xlabel('Star Rating')
    plt.ylabel('Number of Reviews')
    plt.show()
                            
              
################################################################################################################################
# Section 8
'''
Evaluate the performance of each of the three classification models when applied to data from the other two selected categories.
'''

cafe_on_others = evaluate_a_on_b_and_c("cafe_reviews.txt", "gym_reviews.txt", "restaraunt_reviews.txt")
#Originally gave 'dimension mismatch' since the training dataset fixes the vocabulary.
#Solution is to change to vectorizer.transform instead of vectorizer.fit_transform for b & c.

print('Cafe Classifier on Gym Reviews:\n\n', cafe_on_others[0])
print("Accuracy = %.2f \n\n" % cafe_on_others[1])

print('Cafe Classifier on Restaraunt Reviews:\n\n', cafe_on_others[2])
print("Accuracy = %.2f\n" % cafe_on_others[3])
print('==========================================\n')

#-----------------------

gym_on_others = evaluate_a_on_b_and_c("gym_reviews.txt", "cafe_reviews.txt", "restaraunt_reviews.txt")

print('Gym Classifier on Cafe Reviews:\n\n', gym_on_others[0])
print("Accuracy = %.2f \n\n" % gym_on_others[1])

print('Gym Classifier on Restaraunt Reviews:\n\n', gym_on_others[2])
print("Accuracy = %.2f\n" % gym_on_others[3])
print('==========================================\n')

#-----------------------

restaraunt_on_others = evaluate_a_on_b_and_c( "restaraunt_reviews.txt", "cafe_reviews.txt", "gym_reviews.txt")

print('Restaraunt Classifier on Cafe Reviews:\n\n', restaraunt_on_others[0])
print("Accuracy = %.2f \n\n" % restaraunt_on_others[1])

print('Restaraunt Classifier on Gym Reviews:\n\n', restaraunt_on_others[2])
print("Accuracy = %.2f\n" % restaraunt_on_others[3])
print('==========================================\n')


