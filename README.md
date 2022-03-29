# semg-classifier

Semg-classifier contains the Python code for my final year project, Machine Learning for Prosthetic Devices. It contains two files for dealing with pre-exisiting semg data from the [ninapro](http://ninapro.hevs.ch/) (not included as not personal own data). It also contains files dealing with the live collection, processing and position classification of data from semg reader - a [myo armband](https://www.robotshop.com/uk/myo-gesture-control-armband-black.html). This project was developed on the Spyder IDE.

## Installation

Uses the package manager [anaconda](https://www.anaconda.com/) and the anaconda prompt to install the [myo-python](https://github.com/NiklasRosenstein/myo-python) library.

```bash
conda activate spyder-env
pip install myo-python
```

Repeat this process for all other libraries not included in your IDE version.

## Usage

Run feature_extraction.py and change 
```python
annots = loadmat('data\s1\S1_A1_E3.mat')
```
To the location in your file path to ninapro's data or to your own, similiarly formatted, data.

## Contributing
Pull requests are welcome. For major changes or additional information, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
