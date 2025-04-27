CaliStack

CaliStack is a machine learning project that uses smartphone IMU data to detect and count calisthenics movements, like pushups, situps, and squats. By analyzing acceleration data, it helps athletes focus on their workout while automatically tracking their reps.

Repository Structure
Model Training Folder: This folder contains the main code and everything we did throughout the core of the project, including model training scripts and configurations.

Dataset Folder: This folder includes the dataset we used and how we polished the data to get it ready for analysis and processing.

Implementation Folder: This folder shows how we integrated everything together, using an app to stream the accelerometer data, where our Python code then uses the trained model to identify the movements.

UI Folder: This folder contains the user interface, where users can upload their data and run it through the model to identify movements.

Note

Unfortunately, the project didn't go exactly as planned. Due to the many moving parts—such as rep counting, model identification, data streaming, and text-to-speech—while each component worked fine individually, integrating them all together was more complex than anticipated.
Additionally, the limited amount of data collected, due to the tedious nature of the process and the short time available, meant that the model couldn't detect movements as accurately as intended.
