# Music Genre Classification: PCA and Logistic Regression

## Overview

This project focuses on classifying different music genres based on several audio features using Principal Component Analysis (PCA) and Logistic Regression. The goal is to build a model that can predict the genre of a music track by extracting meaningful features and reducing dimensionality using PCA before feeding the data to a logistic regression classifier.

## Project Structure

This repository consists of the following files:

- **Music Genre Classification - PCA and Logistic Regression.ipynb**: Jupyter notebook containing the data exploration, preprocessing, modeling, and evaluation steps.
- **music_dataset_mod.csv**: Dataset containing multiple audio features for different music tracks, along with their corresponding genres.

## Dataset

The dataset (music_dataset_mod.csv) includes the following features:

- **Tempo**: Speed of the music track in beats per minute.
- **Dynamics Range**: Measure of loudness variation.
- **Vocal Presence**: Indicator of vocals in the track.
- **Percussion Strength**: Intensity of percussion elements.
- **String Instrument Detection**: Detection of string instruments in the track.
- **Electronic Element Presence**: Presence of electronic components.
- **Rhythm Complexity**: Complexity of the rhythm pattern.
- **Drums Influence**: Influence of drums in the track.
- **Distorted Guitar**: Presence of distorted guitar sounds.
- **Metal Frequencies**: Frequencies typical for metal music.
- **Ambient Sound Influence**: Influence of ambient sounds.
- **Instrumental Overlaps**: Amount of instrumental overlaps in the track.
- **Genre**: Target variable, indicating the genre of the music track (e.g., Country, Classical, Rock, Hip-hop).

## Project Workflow

The project involves the following steps:

1. **Data Exploration**: The dataset is loaded and inspected to understand the structure, data types, and summary statistics of each feature.
   ```python
   music = pd.read_csv('music_dataset_mod.csv')
   music.head()
   ```

2. **Data Preprocessing**:
   - Encoding categorical labels (the Genre column) using `LabelEncoder`.
   - Standardizing numerical features to ensure uniform scale.
   ```python
   label_encoder = LabelEncoder()
   music_copy['Genre'] = label_encoder.fit_transform(music_copy['Genre'])
   scaler = StandardScaler()
   scaled_features = scaler.fit_transform(music_copy.drop('Genre', axis=1))
   ```

3. **Dimensionality Reduction using PCA**:
   - Applying PCA to reduce the number of features while retaining most of the variance in the data.
   ```python
   pca = PCA(n_components=5)
   pca_features = pca.fit_transform(scaled_features)
   ```

4. **Modeling with Logistic Regression**:
   - Using `LogisticRegression` to classify the music genres based on the reduced set of features obtained from PCA.
   - Splitting the dataset into training and testing sets, and fitting the logistic regression model.
   ```python
   X_train, X_test, y_train, y_test = train_test_split(pca_features, music_copy['Genre'], test_size=0.2, random_state=42)
   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

5. **Model Evaluation**:
   - Evaluating the model using accuracy scores and classification reports.
   - Plotting confusion matrices to visualize the model's performance.
   ```python
   y_pred = model.predict(X_test)
   print(classification_report(y_test, y_pred))
   print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
   ```

## Models Used

### Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that transforms the original features into a smaller set of uncorrelated components that retain most of the variance present in the data. In this project, PCA is used to reduce the number of features from the original dataset to a manageable size, which helps simplify the logistic regression model and reduce the risk of overfitting.

- **Purpose**: Reduce dimensionality, retain maximum variance, and simplify the dataset.
- **Components**: The number of components is set to 5, balancing variance retention and model simplicity.

### Logistic Regression

Logistic Regression is a supervised learning algorithm used for binary and multi-class classification tasks. In this project, it is used to classify music genres based on the features extracted after applying PCA.

- **Purpose**: Classify music tracks into different genres.
- **Advantages**: Logistic regression is simple, efficient, and interpretable, making it suitable for initial experimentation.
- **Training and Testing**: The dataset is split into training and testing sets (80% training, 20% testing), and the model is trained on the training set and evaluated on the test set.

## Visualization

Visualizations play a key role in understanding the data, the transformation it undergoes, and the performance of the model. The following visualizations were created in the project:

- **Correlation Heatmap**: Shows the correlation between different features to identify redundant features that can be removed before applying PCA.
  ```python
  plt.figure(figsize=(10, 8))
  sns.heatmap(music_copy.corr(), annot=True, cmap='coolwarm')
  plt.title('Feature Correlation Heatmap')
  plt.show()
  ```

- **PCA Explained Variance Plot**: Visualizes the amount of variance explained by each principal component.
  ```python
  plt.figure(figsize=(8, 5))
  plt.plot(range(1, 6), pca.explained_variance_ratio_, marker='o', linestyle='--')
  plt.xlabel('Principal Component')
  plt.ylabel('Variance Explained')
  plt.title('PCA Explained Variance')
  plt.show()
  ```

- **Genre Distribution**: Displays the distribution of different music genres in the dataset to understand class imbalance.
  ```python
  plt.figure(figsize=(8, 5))
  sns.countplot(x='Genre', data=music_copy)
  plt.title('Genre Distribution')
  plt.xlabel('Genre')
  plt.ylabel('Count')
  plt.show()
  ```

- **Confusion Matrix**: A confusion matrix to visualize the performance of the logistic regression model on the test data.
  ```python
  from sklearn.metrics import ConfusionMatrixDisplay
  ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
  plt.title('Confusion Matrix')
  plt.show()
  ```

## Results

The logistic regression model, after applying PCA for dimensionality reduction, demonstrates reasonable accuracy in predicting the genre of music tracks based on the given audio features. Evaluation metrics, such as precision, recall, and F1-score, are provided to analyze the model's performance in detail.

## Dependencies

The project uses the following libraries:

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For data preprocessing, PCA, model training, and evaluation.
- **Matplotlib & Seaborn**: For data visualization.

Make sure to install these dependencies before running the notebook:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/music-genre-classification.git
   ```

2. Navigate to the project directory:
   ```bash
   cd music-genre-classification
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Music Genre Classification - PCA and Logistic Regression.ipynb"
   ```

4. Follow along with the notebook to explore the data, preprocess it, train the model, and evaluate the results.

## Future Improvements

- **Feature Engineering**: Further engineering of features such as melody extraction, harmonic analysis, or time series-based analysis could potentially improve the model's accuracy.
- **Advanced Models**: Exploring more advanced classification algorithms, such as Random Forests, Support Vector Machines, or Neural Networks, to compare their performance with logistic regression.
- **Hyperparameter Tuning**: Applying techniques such as Grid Search or Random Search to optimize the hyperparameters of logistic regression or other models.

## Conclusion

This project demonstrates the effectiveness of combining PCA with logistic regression to classify music genres based on audio features. By reducing the dimensionality of the data, we were able to simplify the model while still achieving reasonable predictive performance. Future work can focus on improving feature engineering, experimenting with advanced machine learning models, and optimizing hyperparameters to further enhance classification accuracy and generalizability to other datasets.

## Contributing

Contributions are welcome! If you'd like to improve the project or fix issues, feel free to open a pull request.

## Contact

For questions or suggestions, please contact [Kaushik Rohida](mailto:rohidakaushik@gmail.com).