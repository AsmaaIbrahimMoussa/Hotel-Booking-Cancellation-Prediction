# Hotel Booking Cancellation Prediction

## Overview
This project is a **Flask-based web application** that predicts whether a hotel booking will be **cancelled or not** based on user input. The machine learning model used in this application is a **K-Nearest Neighbors (KNN) classifier** trained on a preprocessed dataset.

## Features
- Web interface for users to input booking details.
- Machine learning model trained with **PCA for dimensionality reduction** and **SMOTE for handling imbalanced data**.
- Backend built using **Flask**.
- Predicts if a booking is **Cancelled** or **Not Cancelled**.

## Dataset
The dataset used for training is **`first inten project.csv`**. It contains various booking details, such as:
- Number of adults & children
- Number of weekend & week nights
- Type of meal plan, room type, and market segment
- Lead time, average price, and special requests
- Reservation date details
