# PV Fault Prediction

## Requirements
- NumPy
- Pandas
- Sklearn

## Prepare Dataset
First, run ```covert.py``` to covert the 'xlsx' data file to 'csv' format.
To merge all csv file, call the ```merge_csv``` function.  

## Run Experiments
Select the csv dataset in the ```main.py``` and run the scripts
to start training and testing.

Current implemented algorithm:
- Multilayer perceptron
- Random forrest 