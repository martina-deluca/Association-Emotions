# Association-Emotions
Project assignment for Natural Language Processing.

The primary objective of this project is to utilize a random set of texts using Social Media APIs to draw the association probabilities between two or more kinds of emotions. This involves applying Sentiment Analysis on the data for recognizing the emotion exhibited. Statistics and Machine Learning techniques are then used on the labeled emotions to identify the association rules. The results denote the probabilities by which one type of sentiment is likely to produce other kinds.

This repository includes the Python Notebook with the code implementation, the source files and the report submitted for the assignment. Source files are:
- **Code Files:**
  -	Reddit_Emotional_Analysis.ipynb: Notebook has 3 parts: the api calls, the Reddit data preprocessing with the emotion recognition and visualizations.
    - Part 1 requires no extra data, it stores the results in csv files.
    - Part 2 requires the files `consol_reddit_comments.csv`, `redditor_info_consol.csv` and it creates `consol_reddit_emotion.csv`.
    - Part 3 requires `consol_reddit_emotion.csv`, `tokenizer.pickle` and `emotion_model.h5`.
  -	Apriori_Algorithm.ipynb: Python notebook that includes the code for the implementation of the apriori algorithm. Takes as input the file `consol_reddit_emotion.csv` and generates the association rules.
  -	FPGrowth_Algorithm.ipynb: Python notebook that includes the code for the implementation of the frequent pattern growth algorithm. Takes as input the file `consol_reddit_emotion.csv` and generates the association rules.
- **Data Files:**
  -	consol_reddit_comments.csv: Input file for emotional analysis notebook.
  -	redditor_info_consol.csv: Input file for emotional analysis notebook.
  -	consol_reddit_emotion.csv: File that contains the results of the sentiment analysis algorithm and is used as input for the association rules algorithms.
- **Additional Files:**
  -	tokenizer.pickle, emotion_model.h5: Additional files required for running the notebook for emotion analysis.
