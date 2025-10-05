🎭 Sentiment Analyzer Web App  
A beginner-friendly **Machine Learning project** built to classify IMDB movie reviews as **Positive 😊** or **Negative 😞** using **Naive Bayes algorithms**.  
This project focuses on practicing text classification concepts and deploying a simple web app using **Streamlit**.  
Although this uses text data, it’s not a deep NLP project — it’s mainly an introduction to **sentiment analysis and model comparison**.

🌐 Try the Web App  
You can try the live app here:  
👉 [Sentiment Analyzer Web App](https://aisolutionsbyhassan-sentiment-analyzer-app-othedk.streamlit.app/)

---

📌 **Features**
- Cleaned and preprocessed IMDB dataset for text classification  
- Vectorized text using **CountVectorizer (Bag-of-Words)**  
- Trained and compared multiple **Naive Bayes models**:  
  - **MultinomialNB**  
  - **BernoulliNB**  
  - **GaussianNB**  
- Deployed using **Streamlit** for real-time text input and prediction  
- Simple, user-friendly interface to test custom reviews  

---

📊 **Model Performance**
| Model | Train Accuracy | Test Accuracy |
|:------|:----------------:|:--------------:|
| **BernoulliNB** | 85.46% | 84.36% |
| **MultinomialNB** | 85.20% | 84.28% |
| **GaussianNB** | 74.14% | 71.32% |

✅ The **BernoulliNB** model performed best overall and was selected for deployment.  
The train-test accuracy gap is very small → indicating **no overfitting** and **good generalization**.

---

⚙️ Tech Stack  
Python, Streamlit, scikit-learn, Pandas, NumPy, NLTK
