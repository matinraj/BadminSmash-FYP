## BadminSmash

This is my Final Year Project which uses computer vision and machine learning to predict the shot styles of a player in a badminton match video.

The aim of this project is to develop a software that is able to provide meaningful statistics and information on match play for badminton enthusiasts.

For more detailed insights, please refer to `Final Project Report.pdf` inside the Documents folder. You may also look at the `Software Test Report.pdf` to know more on the testing procedures done.

**Note:**

Due to current dependency conflicts and deprecation, some features like shot style classification counter and report generation are not working optimally and have been removed.

## Installation

```plaintext
  Since the prediction model is large in size, I have uploaded the source code as a release.
  Please go to the release tab and download the 'badminsmash.source.code.zip' file.
  This zip file contains 2 folders: MCS13-badminsmash and posedetection

  The main project code is inside the `MCS13-badminsmash` folder, where the application should be run from.
  The `posedetection` folder contains the code for prediction model training and testing.

  For a thorough guide on installation and how to use the software, please refer to `User Guide.pdf` in the Documents folder.
  Do note that you need to have certain libraries installed (eg. tensorflow, keras, etc.) in order to run the software.
```

## Prediction

Do make use of the videos provided in the `Videos` folder to run the prediction on. Some are already pre-processed with shuttlecock prediction for your convenience.
