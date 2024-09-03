To run this program, you need Python 3.6 or later bundled with the tkinter package and pip. 
If you do not have pip, you may install using the [sudo apt-get install python3-pip] command on your terminal.

For more detailed installation and software operation, please refer to the user guide.

To install the necessary packages, run [pip install -r requirements.txt] on your terminal

The compiled.py file contains integrated code that runs TrackNet and Shot styles prediction.
The analyse.py file contains code developed by the team that runs shot styles prediction only.
The TrackNet3.py file contains code that runs the shuttlecock and its trajectory tracking only.

To run the program, use [python3 compiled.py] command.

In the analyse.py and compiled.py files, the min_detection_confidence and min_tracking_confidence variables in analyse() function determine the minimum confidence rate of the prediction needed before proceeding to annotate the video. By default, it is set to 0.5

