# Machine Learning Portfolio

A collection of machine learning projects completed as part of the University of Cambridge Data Science programme.

## Projects

### 1. [Customer Segmentation](customer-segmentation/)

Segmented ~103,000 customers from Instacart order data into three distinct groups using K-Means clustering. Used the Elbow Method, Silhouette Analysis, and Agglomerative Hierarchical Clustering (Ward linkage) to determine the optimal number of clusters. Visualised clusters using PCA and t-SNE. Identified three segments: high-frequency/high-value customers, dormant but high-margin buyers, and a large core customer base, with targeted business recommendations for each.

**Methods:** K-Means, Agglomerative Hierarchical Clustering, PCA, t-SNE, StandardScaler

---

### 2. [Predict Dropout Rate](predict-dropout-rate/)

Binary classification to predict student dropout using demographic, academic, and macroeconomic features. Compared Logistic Regression, Random Forest, XGBoost, and a Keras neural network. XGBoost achieved the best performance with 88.7% accuracy and 0.937 AUC-ROC after hyperparameter tuning via GridSearchCV. Key predictive features were curricular units approved/grades, tuition fee status, and scholarship status.

| Model | Accuracy | AUC | F1 (Dropout) |
|-------|----------|-----|--------------|
| Logistic Regression | 88.2% | 0.917 | 0.84 |
| Random Forest | 88.6% | 0.917 | 0.84 |
| **XGBoost (tuned)** | **88.7%** | **0.937** | **0.85** |
| Neural Network | 88.2% | 0.915 | 0.84 |

**Methods:** XGBoost, Logistic Regression, Random Forest, Keras Neural Network, GridSearchCV, MinMaxScaler

---

### 3. [NLP Topic Modelling](nlp-topic-modelling/)

Analysed ~6,300 negative PureGym customer reviews from Google and Trustpilot to identify recurring complaint themes. Applied BERTopic (BERT embeddings + UMAP + HDBSCAN), LDA (Gensim), emotion classification (BERT-base-uncased-emotion), and Falcon-7B-Instruct for LLM-based topic extraction. Identified key complaint themes: staff rudeness, equipment shortages, dirty facilities, gym access issues, poor air conditioning, parking problems, and class cancellations. Concluded that the most cost-effective improvements are staff training and increased cleaning frequency.

**Methods:** BERTopic, LDA, BERT Emotion Classification, Falcon-7B-Instruct, NLTK, WordNet Lemmatization

---

### 4. [Time Series Forecasting](time-series-forecasting/)

Forecasted weekly retail book sales for *The Alchemist* and *The Very Hungry Caterpillar* using Nielsen BookScan data, comparing statistical, ML, and hybrid approaches. Used PCHIP interpolation to handle pandemic-era data gaps. Applied STL decomposition, ACF/PACF analysis, and tested SARIMA, XGBoost, LSTM, and two hybrid architectures (sequential and parallel). XGBoost achieved the best result for *The Very Hungry Caterpillar* (7.98% MAPE), while a hybrid SARIMA+LSTM model performed best for *The Alchemist*.

| Model | The Alchemist (MAE) | Very Hungry Caterpillar (MAE) |
|-------|---------------------|-------------------------------|
| SARIMA | 1,067 | 1,761 |
| **XGBoost** | 1,030 | **616** |
| LSTM | 1,641 | 1,494 |
| **Hybrid Sequential** | **1,022** | 3,377 |

**Methods:** SARIMA (Auto-ARIMA), XGBoost, LSTM, Hybrid Sequential/Parallel, STL Decomposition, PCHIP Interpolation

---

### 5. [Employer Project](employer-project/)

Group project for the Bank of England investigating whether NLP-based sentiment analysis of JPMorgan Chase earnings call transcripts can serve as a leading indicator for financial metrics (Net Interest Income and ROTCE). Built an automated pipeline to extract structured data from 16 quarterly earnings PDFs (Q4 2020 -- Q3 2024) using pdfplumber, then applied five sentiment models: three FinBERT variants, Financial RoBERTa, and GPT-4o. Found that GPT-4o produced the most differentiated sentiment signal, though correlations with actual metric movements remained weak (r = 0.131 forward-looking), suggesting earnings call sentiment alone is insufficient as a predictive indicator.

**Methods:** pdfplumber, FinBERT, Financial RoBERTa, GPT-4o, Pearson Correlation, NLTK, WordNet Lemmatization

---

## Tools and Technologies

Python, Jupyter Notebooks, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow/Keras, NLTK, Gensim, BERTopic, Hugging Face Transformers, OpenAI API, pdfplumber, Matplotlib, Seaborn, pyLDAvis
