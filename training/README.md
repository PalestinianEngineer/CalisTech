Here you can find the main code and scripts for everything we did, from preprocessing the IMU data to training the model.
We trained multiple models:

an MLP model (great for sensor data and low latency),

a random forest algorithm (also low latency),

and a CNN-LSTM model (trained locally because Colab couldn't be used for real-time data streaming).

We also used the sections of the dataset containing 10 reps of pushups and situps to train the model to automatically count repetitions.
The code is well-documented, so you can find all the necessary information and instructions within the scripts themselves.

Colab file:
https://colab.research.google.com/drive/1NaitBNEhaHL5JFVz341cX5ZPWFmupwcQ?usp=sharing 
