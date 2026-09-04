# Email Spam Detection

Binary NLP classifier distinguishing spam from legitimate ("ham") emails, using TF-IDF feature extraction and two classifiers: Multinomial Naive Bayes and Logistic Regression.

## Dataset

Source: [Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset) (enron-1 subset), 5,171 emails, ~71% ham / 29% spam.

Note on dataset selection: an earlier candidate dataset (`marcelwiechmann/enron-spam-data`, the full Enron-Spam corpus) was rejected after inspection revealed a labeling quality issue — the spam class consisted almost entirely of duplicated legitimate business correspondence rather than genuine spam content. This subset was verified for label integrity (no text/label conflicts, realistic class balance, spam samples confirmed as actual promotional/scam content) before use.

## Approach

Duplicate emails were dropped after confirming no label conflicts existed within duplicate groups. Text was lowercased, stripped of punctuation and numbers, filtered for stopwords, and lemmatized. Features were extracted with TF-IDF (5,000 features, fit on training data only to prevent leakage). Two classifiers were trained: Multinomial Naive Bayes as the standard text-classification baseline, and Logistic Regression as the alternative. Both were evaluated on accuracy, precision, recall, F1, and confusion matrix.

## Results

| Model | Accuracy | Spam Precision | Spam Recall | False Negatives | False Positives |
|---|---|---|---|---|---|
| Naive Bayes | 0.95 | 0.89 | 0.96 | 13 | 33 |
| Logistic Regression | 0.98 | 0.94 | 0.98 | 5 | 17 |

Logistic Regression performed better on every metric that matters here, missing fewer spam emails and misclassifying fewer legitimate emails as spam.

## Known limitations

Numeric content (prices, phone numbers) was stripped during preprocessing and may have discarded genuine spam signal. TF-IDF was capped at 5,000 features against roughly 37,600 unique terms in the training vocabulary, dropping the long tail. Lemmatization used default POS tagging, so some verbs weren't reduced to their true root form.

## Tech stack

Python, pandas, scikit-learn, NLTK, matplotlib, seaborn, WordCloud
