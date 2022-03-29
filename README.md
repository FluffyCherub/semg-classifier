# semg-classifier

Semg-classifier contains the Python code for my final year project, Machine Learning for Prosthetic Devices. It contains files for processing pre-exisiting semg data from the [ninapro](http://ninapro.hevs.ch/) (not included as not my own personal data). It also contains files dealing with the live collection, processing and position classification of data from semg reader - a [myo armband](https://www.robotshop.com/uk/myo-gesture-control-armband-black.html). This project was developed on the Spyder IDE and some code was taken directly from [shayanalibhatti's Finger-Movement-Classification...](https://github.com/shayanalibhatti/Finger-Movement-Classification-via-Machine-Learning-using-EMG-Armband-for-3D-Printed-Robotic-Hand).

## Installation

Uses the package manager [anaconda](https://www.anaconda.com/) and the anaconda prompt to install the [myo-python](https://github.com/NiklasRosenstein/myo-python) library.

```bash
conda activate spyder-env
pip install myo-python
```

Repeat this process for all other libraries not included in your IDE version.
Myo-python uses the sdk provided, in the past, by Thalmic Labs and now made available in myo-python's repository in the [releases](https://github.com/NiklasRosenstein/myo-python/releases) section to connect the myo armband.

## Usage

### Machine Learning Practice
Run feature_extraction.py and change 
```python
annots = loadmat('data\s1\S1_A1_E3.mat')
```
To the location in your file path to ninapro's data or to your own, similiarly formatted, data.


### Live data
#### initial_live_data_recording
Run initial_live_data_recording.py. This currently records movements for ~500 seconds, expecting 5 movements; a pause in the rest position for 5 seconds, the position for 5 seconds, repeated 10 times for each movement. Change 
```python
number_of_samples *= 50 # About 100 seconds
generate_stimulus
```
To reflect your own needs or patterns. This outputs the unprocessed labelled (0:rest, 1-5 other positions) emg data as a .csv file and a graph of the data (which can be commented out if you like).

#### process_raw_emg_data
Run process_raw_emg_data.py. This performs feature extraction (windowing and RMSing the data) and tests the data using the K-Nearest Neighbour Classifier and using the Multi Level Perceptron from [sklearn](https://scikit-learn.org/stable/).
Change 
```python
period = 400 
```
To change the windowing period.
Change 
```python
model = KNeighborsClassifier(n_neighbors=2)
clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(65,), random_state=1,learning_rate_init=0.005,max_iter=500)
```
To change the classifiers' parameters to fit your own data better. 
This code outputs the labelled RMS data as a .csv file and a graph of the stimulus and RMS blocks (which can be commented out if you like).

#### classify_live_data
Run classify_live_data.py. This performs live classification based on the data recorded as above using the K-Nearest Neighbour Classifier. This will run forever or until you press ctrl+c in the command line and outputs a graph of the classification (as integers) in real time using [Matplotlib](https://matplotlib.org/). 
In the future this will connect to an [arduino](https://www.arduino.cc/) device to demonstrate effectiveness for prosthetics.

## Contributing
Pull requests are welcome. For major changes or additional information, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
