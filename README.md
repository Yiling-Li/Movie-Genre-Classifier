Movie Genre Classifier
===


## Table of Contents
---
- [Movie Genre Classifier](#movie-genre-classifier)
  * [Table of Contents](#table-of-contents)
  * [Introduction](#introduction)
  * [Model](#model)
    + [Model Architecture](#model-architecture)
    + [Parameters in the Model](#parameters-in-the-model)
    + [Model Performance](#model-performance)
  * [Data](#data)
    + [Data Source](#data-source)
    + [Data Summary](#data-summary)
    + [Data Transformation](#data-transformation)
    + [Data Split](#data-split)
  * [Training](#training)
    + [Training curve of Model](#training-curve-of-model)
    + [Hyper-parameters Tuning](#hyper-parameters-tuning)
  * [Results](#results)
    + [Quantitative Measurements used for Evaluation](#quantitative-measurements-used-for-evaluation)
    + [Quantitative results](#quantitative-results)
    + [Qualitative results](#qualitative-results)
    + [Justification for Model Performance](#justification-for-model-performance)
  * [Ethical Consideration](#ethical-consideration)
    + [Limitation of Data](#limitation-of-data)
    + [Limitation of Model](#limitation-of-model)
  * [Authors](#authors)

## Introduction
---
Our model aims to perform sequence classification on movie plot summaries, sorting them into their correct genres (horror, comedy, documentary, etc.). We will achieve this through the use of an bi-directional LSTM model.

## Model
---

### Model Architecture
We use a bi-directional LSTM (Long Short Term Memory) model with GloVe embeddings. The LSTM is similar to a traditional RNN in the sense that they both pass forward a hidden state based on past computations. The difference is that the LSTM also carries forward a “cell state” which contains words that the model deems important to correctly understand the context of a given plot summary, no matter how early in the sentence they were observed. This cell state is managed by the forget (determines how much information to “forget” in the cell state) and input (determines how much new information we update the cell state with) gates. As shown in the diagram, there are two forward passes: one that inputs the summary in its original sequence, and one that inputs the summary in reverse.

![](https://i.imgur.com/cTPfJW5.png)

### Parameters in the Model
Number of parameters in fc layer = `(hidden_size*2)* number_of_outputs + number_of_outputs`
`= (60*2)*5 + 5 = 121*5 = 605`

Since a PyTorch LSTM has 4 neural network layers, and 2 bias vectors our formula for the number of parameters is:   

Number of parameters in LSTM = `4 * ((input_size+2) * hidden_size + hidden_size^2)`
Number of parameters in BiLSTM = `Number of parameters in LSTM *2`
`= 4 * ((50+2) * 60 + 60^2) *2`
`= 8 * (52*60 +60^2)`
`= 53760`

Total number of paramters = 53760 + 605 = 54365


### Model Performance
Example of successful classification:

    True label: Thriller

    Movie name: The China Syndrome

    Plot: 
    While visiting the ventana nuclear power plant outside los angeles  television news reporter kimberly wells   her maverick camerman richard adams  and their soundman hector salas witness the plant going through an emergency shutdown . shift supervisor jack godell  notices an unusual vibration while grabbing his cup of coffee which he had set down; then he finds that a gauge is misreading and that the coolant is dangerously low . the crew manages to bring the reactor under control. richard surreptitiously films the incident  despite being requested to not film *the control room....* (truncated)

    Predicted label: Thriller

Example of unsuccessful classification:

    True label: Crime Fiction
    
    Movie name: The Woman Knight of Mirror Lake
    
    Plot: 
    The film tells the story of qiu jin and her involvement in revolutionary uprisings against the qing dynasty in anhui province. influences on her life are shown through a series of flashbacks. as a child  qiu jin resisted having her feet bound according to common practice  and instead pursued her interests to learn horse riding  martial arts and literature with her father and brother. through her poetry  she expresses the her sorrow at the weak state of the nation and the repression of women. finding other like minded women in beijing and then travelling to japan to study reinforces her view that nationalist action is required to reform china. returning to china  qiu jin takes the position as xu xilin's lieutenant  assisting with the training of revolutionaries at the datong school and plotting the revolution. xu xilin is later captured while executing the assassination of the governor  and qiu jin is captured when government forces storm the datong school. qiu jin is tortured in an attempt to reveal other conspirators and she is later executed.

    Predicted label: Short Film
## Data
---

### Data Source

We used data from the CMU Movie Summary Corpus (http://www.cs.cmu.edu/~ark/personas/)
collected by David Bamman, Brendan O'Connor, and Noah Smith at the Language Technologies
Institute and Machine Learning Department at Carnegie Mellon University. The data is released
under a Creative Commons Attribution-ShareAlike License:
https://creativecommons.org/licenses/by-sa/3.0/us/legalcode.

### Data Summary
The CMU Movie Summary Corpus dataset contains 42 306 movie plot summaries extracted from Wikipedia, as well as corresponding metadata extracted from Freebase. The metadata includes the movie’s genres and Wikipedia movie ID, among other information. We only used the genre information, Wikipedia movie ID, plot summary, and movie title for this project. 
For the project we focused on classifying movies from the 5 most frequent genres in the dataset. These genres are Thriller (5339 occurrences), Drama (5289 occurrences), Crime Fiction (2928 occurrences), Short Film (2919 occurrences), and Romantic Comedy (1930 occurrences). We decided to use 500 movies from each genre, so our dataset has a total of 2500 datapoints. In addition, each 500-movie genre makes up 20% of the 2500-sized dataset. 
Across the 2500 plot summaries there are 678747 words in total, and there is an average of about 271.5 words per plot summary. 

### Data Transformation
First, the plot summary data was read in from plot_summaries.txt. The plot summaries were read in and processed one by one to remove text that was not relevant to the plot summary (e.g. tags such as {{Expand Section}}, links, metadata about the Wikipedia article). We also removed punctuations marking the end of a sentence (e.g. periods, question marks, exclamation marks), as well as single and double quotes. We removed the sentence-ending punctuations because we wanted our plot summaries to be seen by the model as one single sentence, and we removed the quotation marks to extract the words that had quotation marks attached (e.g. to extract hello from “hello”). Finally we removed all leading and trailing whitespaces, and made all words lowercase since GloVe only contains lowercase words.
	After this step, we removed the plot summaries with below 8 words from the dataset. We chose 8 as our minimum word count because we found it was the minimum length for plot summaries where we (i.e. humans) could still predict the genre. We stored the plot summaries that were long enough in a list. Each plot summary was stored with the corresponding movie ID.
	Next we used GloVe to convert each string word in each plot summary into their corresponding index in GloVe. While doing this we ignore plot summaries where none of the words have an embedding in GloVe. Using these transformed plot summaries, we created a modified dataset where each element contains the movie ID and the transformed plot summary.

Now we read in the movie metadata from movie.metadata.tsv. For each movie, we extracted the genre information. Most movies in the dataset had multiple genres, however we only wanted to use 1 genre label per movie. So, we extracted the first genre listed for each movie to serve as the label.
Since we only want movies from the top 5 genres, we created a dictionary that mapped each string genre to a unique integer, ranging from 0-4. These are the integer representations of the genre labels. Using movies only from the top 5 genres, we created our final dataset, where each element contains the integer genre label, the GloVe transformed plot summaries, and the movie ID.

### Data Split
For each genre, we divided 60% of the data for the training set, 20% of the data for the validation set, and 20% of the data for the test set. So our training set would contain 300 movies from each genre, for a total of 1500 movies. Each genre made up 20% of the training set.  Both the validation and test set would contain 100 movies for each genre, for a total of 500 movies each. Each genre made up 20% of the test and validation set. We decided to split within each genre because we wanted all 3 sets to have movies from each genre, and to have a similar distribution of genres as our entire dataset. For example, we didn’t want the training set to contain 0 Thriller movies, while the test and validation set contained 250 Thriller movies each. If we did this, our model would not learn to predict Thriller movies at all, even though they make up 20% of the dataset.




## Training
---

### Training curve of Model

Top 5 genres model:
![](https://i.imgur.com/2LlIlbY.png)

![](https://i.imgur.com/mU9luJw.png)

### Hyper-parameters Tuning
5 genres:
* For hyperparameter batch_size= 1,weight_decay=0, learning_rate=0.1, num_epochs=10, the improvement of training accuracy and validation accuracy is small and both of them end with a low accuracy. The model tends to underfitting to the data, with only 62% as training accuracy and 25% as validation accuracy. It does not capture enough features of the data and the model is quite simple.
* For hyperparameter batch_size= 1,weight_decay=0, learning_rate=0.001, num_epochs=10, the improvement of training accuracy is large, but not for the validation accuracy. The model tends to overfitting to the data, with 100% as training accuracy but only 30% as validation accuracy. It capture too many features of the training dataset and the model is complex and does not generalize. There is a quite large difference between training and validation accuracy. 

## Results
---

### Quantitative Measurements used for Evaluation

The quantitative measure used to assess the result was a simple Classification Accuracy, given by:
\begin{aligned}
 \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total Number of predictions}} 
\end{aligned}
This measure is a reasonable method to evaluate a classification problem with an equal number of data belonging to every class; in the Data Pre-Processing step, it is mentioned that the data used for this project takes care of imbalances in classes.

The model gives a probability distribution of the prediction for a particular data point, and we choose the prediction with the highest probability as our prediction. The prediction is then compared to the actual class; if it is correct, the total number of correct predictions is incremented by one.

### Quantitative results

| Dataset     | Accuracy    |
| ----------- | ----------- |
| Train       |  97.1%      |
| Validation  |  39.6%      |
| Test        |  35.0%      |

Since the objective of our model is to correctly classify the top 5 movie genres based on the plot summary, considering the computing power and the duration of the project, our model has performed reasonably well. As for a new unseen data, our model would correctly classify into a genre based on the plot summary 35% of the time.

### Qualitative results

| Plot summary     | Predicted   | Actual   |
| -----------      | ----------- | -------- |
| Plot 1: The China Syndrome while visiting the ventana nuclear power plant outside los angeles  television news reporter kimberly wells   her maverick camerman richard adams  and their soundman hector salas witness the plant going through an emergency shutdown . shift supervisor jack godell  notices an unusual vibration while grabbing his cup of coffee which he had set down; then he finds that a gauge is misreading and that the coolant is dangerously low . the crew manages to bring the reactor under control. richard surreptitiously films the incident  despite being requested to not film the control room for security purposes. when he shows the film to experts  they realize that the plant came close to the china syndrome in which the core would have melted down into the earth  hitting groundwater and contaminating the surrounding area with radioactive steam. during an inspection of the plant before it's brought back online  a technician discovers a small puddle of radioactive water that has apparently leaked from a pump. godell pushes to delay restarting the plant  but the plant superintendent denies his request and appears willing to let nothing come in the way of the scheduled restart of the plant. godell investigates further and uncovers evidence that radiographs of welds on the leaking pump have been falsified . he believes that the plant is unsafe and could be severely damaged if another full-power scram occurs. he tries to bring the evidence to plant manager herman deyoung  who brushes godell off as paranoid and states that new radiographs would cost at least $20 million. godell confronts d.b. royce  an employee of foster-sullivan  the construction company who built the plant  as it was royce who signed off on the welding radiographs. when godell threatens to go to the nuclear regulatory commission  royce threatens him and a pair of goons from foster-sullivan park outside his house. kimberly and richard confront godell at his home with what they know  at which point  the engineer voices his concern about the vibration he felt during the scram and his anger about the false radiographs. kimberly and richard ask godell if he'll come clean at nrc hearings  being held at point conception  where foster-sullivan is looking to build another nuclear plant. godell tells them about the threat on his life  but agrees to get them  through hector  the false radiographs to take to the hearings. following the exchange though  hector is run off the road by company men and the radiographs are taken. godell leaves for the hearings but is chased by the goons out his house. he escapes by taking refuge in the plant. to his dismay  godell finds that the reactor is being brought up to full power. he grabs a gun from a security guard  forces everyone out  including his best friend and co-worker ted spindler   and demands to be interviewed on live television by kimberly. plant management agrees to the interview  which buys them time as they try to regain control of the plant. minutes into the broadcast  plant technicians deliberately cause a scram so they can retake the control room . while godell is distracted by the scram alarms a swat team forces its way into the control room. the television cable is cut and a panicky godell is shot by the police  but before he dies he feels the unusual vibration again. the resulting scram is only brought under control by the plant's automatic systems. true to godell's predictions  the plant suffers significant damage as the pump malfunctions. plant officials try to paint godell as emotionally disturbed  but ted states that godell would not have taken such drastic steps had there not been something wrong. the film ends with a tearful kimberly concluding her report as the reporter's live signal abruptly cuts to color bars.          | Thriller    | Thriller |
| Plot 2: The Woman Knight of Mirror Lake the film tells the story of qiu jin and her involvement in revolutionary uprisings against the qing dynasty in anhui province. influences on her life are shown through a series of flashbacks. as a child  qiu jin resisted having her feet bound according to common practice  and instead pursued her interests to learn horse riding  martial arts and literature with her father and brother. through her poetry  she expresses the her sorrow at the weak state of the nation and the repression of women. finding other like minded women in beijing and then travelling to japan to study reinforces her view that nationalist action is required to reform china. returning to china  qiu jin takes the position as xu xilin's lieutenant  assisting with the training of revolutionaries at the datong school and plotting the revolution. xu xilin is later captured while executing the assassination of the governor  and qiu jin is captured when government forces storm the datong school. qiu jin is tortured in an attempt to reveal other conspirators and she is later executed.          | Short Film            | Crime Fiction |

The above table is just two examples from the test data; Plot 1 was correctly classified, whereas Plot 2 wasn't by the model. Reading the summary of Plot 1, it is pretty straightforward for a human to understand and classify correctly. Whereas, reading the summary of Plot 2 is confusing as the summary does feel like a Thriller/Crime Fiction but the true label according to dataset was Short Film.

### Justification for Model Performance

Our results weren't the best from the above as they had a low test accuracy percentage. Considering the difficulty of the problem and issues with our dataset and model (described below), our test data performed just as well as the validation data.

Since data trained for the model was limited to 1500 data points, with 300 data points belonging to each of the top 5 genres, there wasn't much data for the model to learn. Although the average text length of a plot summary is about 272, the model would've performed better if it had more training data.

The dataset did not have many variances within the genre, as the top 5 genres: 'Crime Fiction,' 'Romantic Comedy,' 'Drama,' 'Short Films,' 'Thriller'; since most of the films in these genres have a cliche plot except for 'Short Films', and could be a reason why the model overfitted the training set.

Due to the long training time, it was challenging to find hyperparameters to damp overfitting. Furthermore, increasing weight decay resulted in the accuracies having no improvement; this is discussed further below.

With a more robust system and time, this model could be trained on a larger dataset to attain better results than the ones above. As including more data can insert some variance that can prevent the model from overfitting, and a robust system can shorten the code's runtime, so different hyperparameters can be tested and trained.


## Ethical Consideration
---

Our model can be beneficial to streaming service users in finding a movie based on a specific genre, and this can help reduce the time in finding what to watch. Classifying into a genre is also beneficial to filmmakers and producers since it can be marketed better on a target audience like a specific genre. The streaming platform benefits from knowing what genre a user is more likely to watch and use it to give better recommendations.

Despite the benefits this model can provide, it is not perfect without its issues.

### Limitation of Data

Since a model is only as good as the data it is trained upon, it is crucial to examine the training data closely as the improvements in the data can result in a more robust model.

Since the data used for training had only the ten most popular genres from the original dataset, limiting the training set to only the selected genre could result in high misclassification error when a genre that does not belong to the training set is fed to the model to predict.

The plot summary dataset was reduced to having only one single genre as a class label per data, but in reality, it is more common to have a film with multiple genres or transcends genre labels. This is detrimental to filmmakers making genre-bending films as the misclassification could stray away from the targeted audiences.

However, even if we can correctly classify a film to its actual genre with 100% accuracy, two films in the same genre can be widely different based on the era of the movie, the Motion Picture Content rating, the style of filmmaking, or the language of the film. For example, there are sub-genres within comedy, such as "Adult Comedy," which is unsuitable for the younger audience. 

Of course, it is desirable to include all this diversity, have multiple genres associated with a single plot summary, and consider the era of the film, Content rating, and other factors, but this project is out of scope.

### Limitation of Model

Despite Bi-LSTM working well for many problems especially remembering long-term information, it is not perfect. While building the project, we encountered issues that were hard to rectify.

1. The model training time is too long; one of the main issues we encountered was this. Therefore, it was hard to test out different hyperparameters.
2. Overfitting the training data, the model overfits easily. It is also challenging to implement the dropout layer to mitigate this.
3. Bi-LSTM model requires more memory bandwidth to train; therefore, it is computationally inefficient. 


## Authors
---

Nithin Premkumar 1005555436
- Code:
    - Model Creation
    - Results
    - Overfitting
- README:
    - Results
    - Ethical Consideration

Oluwatomiwa Olasoko 1005488928
* Code:
    * Model design and testing
    * Hyperparameter testing
    * Results
* README:
    * Introduction
    * Model

Yiling Li 1004712894
* Code:
    * Embedding
    * Summarys Batcher
    * Loader
    * Overfitting and Training 
    * Hyperparameter Tuning
* README:
    * Training curve
    * Hyperparameter Tuning

Stella Song 1002639380
* Code:
    * Data cleaning and preprocessing
    * Data transformation
    * Dataset splitting
    * Overfitting and Training
* README:
    * Data

