# Fashion Recommendation System

This repository contains a fashion recommendation system that uses deep learning techniques to classify fashion items and recommend similar products based on visual similarity.

## Repository Structure

- `classifier.ipynb`: Jupyter notebook for training the fashion item classifier (requires TPU).
- `recommender.ipynb`: Jupyter notebook for building the recommendation system (requires GPU).
- `app.py`: Streamlit application for user interaction with the system.
- `data/`: Directory containing dataset information.
  - `styles.csv`: Fashion item metadata.
  - `download_dataset.txt`: Instructions to download the full dataset.
- `models/`: Directory for storing trained models.
  - `download_model.txt`: Instructions to download the best classifier model.
- `recommendation/`: Directory containing files for the recommendation system.
  - Various model and data files.
  - `download_pca_features.txt`: Instructions to download PCA features.

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/mohamadsolouki/Fashion-Classifier-and-Recommender.git
   cd Fashion-Classifier-and-Recommender
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the necessary data and model files as instructed in the respective text files in the `data/`, `models/`, and `recommendation/` directories.

## Usage

### Training the Models

Due to high computational requirements, it's recommended to run the training notebooks on Kaggle:

- Classifier: [[Kaggle Notebook Link for classifier.ipynb](https://www.kaggle.com/code/emsaad/comprehensive-classifier-system)]
- Recommender: [[Kaggle Notebook Link for recommender.ipynb](https://www.kaggle.com/code/emsaad/comprehensive-recommendation-system)]

### Running the Streamlit App

To run the Streamlit app locally:

1. Ensure all required files are downloaded and in their correct locations.
2. Run the following command:
   ```
   streamlit run app.py
   ```
3. Open a web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file in the root directory for more details.

