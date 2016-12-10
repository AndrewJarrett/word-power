# Word Power
The purpose of this project is to recreate the *Word Power* paper by Jegadeesh and Wu. There is a documented Jupyter Notebook in `notebooks/Word Power.ipynb` which can be run in order to generate results, but it is not able to use multiprocessing so it is slower than the actual program. To leverage multiprocessing abilities, run the `main.py` program via `$ python main.py`.

## System Requirements
Given that Python and Redis will need to load the data into system memory (RAM) the program can get very memory intense. It is necessary to at least have **16GB of RAM** in order to run the full program (from year 1995 - 2008). In order to run this program, it is necessary to have Redis installed and working properly. If you are using Windows, Redis can be installed through [Chocolatey](https://chocolatey.org/) via `C:> choco install redis-64`. If you are running MacOSX you can install Redis via `$ brew install redis`. If you are running Linux or another Unix you should be able to install Redis through your package manager or compile from source.

## Dependencies
This program depends on having the necessary software and packages in order to run. First, you need to have `Python 3.5.2` installed. Next, you should be able to install all Python software dependencies through running `$ pip install -r requirements.txt`. In order to get the `lxml` package installed on Windows, it may be necessary to install the `.whl` file located in the `lib` project directory via `C:> pip install lib/lxml-3.6.4-cp35-cp35m-win_amd64.whl`. We had some issues with the third-party package `SECEdgar` and had to modify it in order to get it to work properly. Once the package is installed via `pip` it is possible to copy our version in the `lib/SECEdgar` folder and overwrite the version downloaded via `pip` if needed. 

## Project Structure
This outlines the project structure.
* `data` - This folder contains necessary data to run the analysis. The merged CRSP and Compustat datafile is too large to include in the project so it is necessary to run the SAS program (`CRSP+Comp.sas`) to generate the `crsp_comp.sas7bdat` data file first.
* `data/_amended` - This folder is used to hold 10-K files that are amended 10-Ks
* `data/_error` - This folder is used to hold 10-K files that contained errors that made it impossible to analyze
* `data/_nostockdata` - This folder contains 10-K files in which we had no stock information for the company on the filing date
* `data/_outofrange` - This folder contains 10-K files that are outside of our date range that we are looking at

