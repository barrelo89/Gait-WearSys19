# Walk to Show Your Identity: Gait-based Seamless User Authentication Framework Using Deep Neural Network

This is python code for the paper work published in WearSys '19: The 5th ACM Workshop on Wearable Systems and Applications. You can access to the paper through this [link
](https://dl.acm.org/doi/10.1145/3325424.3329666)

## Prerequisities
- Language: Python
- Required Packages: numpy, pandas, matplotlib, scipy, sklearn, tensorflow
- To install the required package, type the following command (To install Tensorflow, visit the official Tensorflow Webpage [link](https://www.tensorflow.org/install)):

1) Python 2
```
pip install numpy pandas matplotlib scipy sklearn
```
2) Python 3
```
pip3 install numpy pandas matplotlib scipy sklearn
```

## Running the code
1. Data Filtering & Visualization of Frequency Distribution
```
python3 data_filter_fft.py
python3 valid_start_end.py
```
![Data Filter](figure/cycle_distribution.png)

2. Gait Cycle Detection: slice walk cycles from the data sequences
```
python3 cycle_detection.py
```
![Interpolation](figure/cycle_length.png)

3. Interpolation: make walk cycles consistent in length
```
python3 interpolation.py
```
![Interpolation](figure/interpolation.png)

4. Cycle Filtering: filter out noisy cycles
```
python3 cycle_filter.py
```
![Cycle Filter](figure/cycle_filter.png)

5. Classification: DNN (Multi Layer Perceptron), CNN, and RNN (LSTM)
```
python3 DNN.py
python3 CNN.py
python3 RNN.py
```
![Authentication](figure/authentication.png)
