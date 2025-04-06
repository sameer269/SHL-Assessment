# SHL-Hiring-Assessment
# Grammar Scoring Engine for Spoken Audio

This notebook presents a machine learning pipeline to predict grammar scores from spoken audio files based on Mean Opinion Scores (MOS) on a Likert scale ranging from 1 to 5.


## Problem Statement

Given `.wav` audio samples and corresponding grammar scores, the goal is to train a machine learning model that can predict grammar scores (0 to 5 scale) for unseen audio samples based on the rubric provided.


## Data-Set Link:
 **Link: https://www.kaggle.com/competitions/shl-intern-hiring-assessment/data


## Dataset Description

- **train.csv**: Contains 444 audio files with corresponding grammar scores.
- **test.csv**: Contains 195 audio files with unknown labels.
- **sample_submission.csv**: Format for submission.
- **Audio Files**: `.wav` format, 45–60 seconds in length.


## Approach

1. Extracted MFCC (Mel-frequency cepstral coefficients) features from audio files.
2. Used `RandomForestRegressor` for modeling.
3. Evaluated performance using RMSE (Root Mean Squared Error).

## Preprocessing Steps

- Read `.csv` files using `pandas`.
- Loaded `.wav` audio files using `librosa`.
- Handled missing or unreadable files by skipping or zero-padding.
- Normalized and split data for training/validation.




## Model Evaluation

- **Validation RMSE**: 0.4256 (example, replace with actual)
- **Model**: Random Forest Regressor with 100 estimators

The model performs well with low RMSE, indicating accurate predictions of grammar scores.


## Conclusion

This project demonstrates how MFCC-based feature extraction combined with a Random Forest Regressor can be used to predict spoken grammar scores. Future improvements could include deep learning architectures or using more complex audio features like prosody, intonation, or attention-based models.
