from PyQt5.QtCore import * 
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import *
from ScorePredictorNCAAB import ScorePredictorNCAAB as SPN
import os

###### Button Functions ######
# Get Path
def get_data_path():
    data_path = QFileDialog.getExistingDirectory(w,"Select Directory to NCAAB Predictor Files")
    spath.setText(data_path)

def get_out_path():
    result_path = QFileDialog.getExistingDirectory(w,"Select Directory for Prediction Results")
    sout_path.setText(result_path)

def make_predictions():
    sstatus.clear()
    predictor = SPN()

    # Get GUI Inputs
    #data_path = in_path
    data_path = spath.text()
    prediction_path = sout_path.text()

    # Make Predictions
    status = predictor.predict_scores(data_path, prediction_path)
    sstatus.setText(status)

def update_model():
    sstatus.clear()
    predictor = SPN()

    # Get GUI Inputs
    data_path = spath.text()

    # Update Model with Current Weeks Data
    predictor_status = predictor.update_model(data_path)
    sstatus.setText(predictor_status)

def reset_model():
    sstatus.clear()
    predictor = SPN()

    start_year = sreset_years1.text()
    end_year = sreset_years2.text()
    model_path = spath.text()

    model_fit = predictor.retrain_model(model_path, start_year, end_year)

    out_string = f"Fit Score: {model_fit}"

    sstatus.setText(out_string)

###### GUI ######

app = QApplication([])

w = QWidget() 
#w.setGeometry(200,200,400,200) 
#############################################################################################################
# Data and Model Path

# Adding a Label for Model Path
sPath_label = QLabel(w)
sPath_label.move(1,10)
sPath_label.setText("Predictor Directory:")

# Adding a Text Entry Box for Model Path
spath = QLineEdit(w)
spath.setGeometry(125, 5, 300, 30) 

# Adding Button to Open A Dialogue Box to Select PDF
Path_btn = QPushButton(w)
Path_btn.setGeometry(430, 2, 200, 35)
Path_btn.setText("Get Predictor Directory")
Path_btn.clicked.connect(get_data_path)

#############################################################################################################
# Prediction Path

# Adding a Label for Prediction Path
sOut_label = QLabel(w)
sOut_label.move(1,50)
sOut_label.setText("Results Directory:")

# Adding a Text Entry Box for Prediction Path
sout_path= QLineEdit(w)
sout_path.setGeometry(125, 45, 300, 30) 

# Adding Button to Open A Dialogue Box to Select PDF
Out_btn = QPushButton(w)
Out_btn.setGeometry(430, 42, 200, 35)
Out_btn.setText("Get Output Directory")
Out_btn.clicked.connect(get_out_path)

#############################################################################################################

# Adding Button to Make Predictions
Prediction_btn = QPushButton(w)
Prediction_btn.setGeometry(1, 125, 200, 35)
Prediction_btn.setText("Make Predictions")
Prediction_btn.clicked.connect(make_predictions)

#############################################################################################################

# Adding Button to Update Model

Update_btn = QPushButton(w)
Update_btn.setGeometry(200, 125, 200, 35)
Update_btn.setText("Update Model Data")
Update_btn.clicked.connect(update_model)

#############################################################################################################

# Status Indicator

sstatus = QLineEdit(w)
sstatus.setGeometry(399, 120, 227, 45) 

#############################################################################################################

# Adding a Label for Model Retraining

reset_label = QLabel(w)
reset_label.move(1,90)
reset_label.setText("Retraining Years:")

# Adding a Text Entry Box for Retraining

sreset_years1 = QLineEdit(w)
sreset_years1.setGeometry(125, 85, 150, 30) 

sreset_years2 = QLineEdit(w)
sreset_years2.setGeometry(275, 85, 150, 30) 

# Adding a Button for Retraining

reset_btn = QPushButton(w)
reset_btn.setGeometry(430, 82, 200, 35)
reset_btn.setText("Retrain Model")
reset_btn.clicked.connect(reset_model)

#############################################################################################################

# Execute GUI
w.show()
app.exec()