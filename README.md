# ProSeries Crash Analysis Tool (CAT)

CAT is a tool for offline crash report analysis. CAT allows faster, more precise queries of ProSeries crash data.  It includes a quickbase crash report downloader, xml parser, Pandas dataframe helper functions, and some text analysis tools.

If you are new to Python 3[<sup>1</sup>](#references), Jupyter notebooks[<sup>2</sup>](#references), or Pandas[<sup>3</sup>](#references), check out the references section.

After installation, check out `CrashAnalysisTour.ipynb` to see examples of the commands available with CAT. Copy and rename `ExampleNotebook.ipynb`
to create a new crash report.

## Installation

### Prerequisite Software

Python 3 is required (3.5+ preferred). We recommend installing python with [Anaconda](https://www.continuum.io/downloads).

[PyCharm](http://jetbrains.com/pycharm) or the [Visual Studio Python Plugin](https://github.com/Microsoft/PTVS/) is recommended for editing the `crash_analysis` library, but not required.

Git is also required for editing the `crash_analysis` library.

### Set Up
In a terminal or command prompt, do the following:

1. Download this repository: `git clone https://github.intuit.com/arosengarten/CrashAnalysisTool.git`. 
If you don't have `git` installed, this can be downloaded from the repo webpage by clicking the "Clone or Download" button and selecting "Download Zip". 
However, if you don't use git/clone the repo, you will not be able to make lasting changes to the tool. 

2. Go inside the directory: `cd CrashAnalysisTool`. If you downloaded the zip file, extract it and go inside that directory. 

3. (Recommended) Create a virtual environment: `conda create --name cat35 python=3.5`. Otherwise, ensure that Python 3.5+ is your default python installation. 

4. (Recommended) Activate the virtual environment (of python 3.5): `source activate cat35` for OSX/Linux, or `activate cat35` for Windows.

5. Install required python packages: `pip install -r requirements.txt`

6. Open or create `crash_analysis/private.py` and input the database id, username, password, and app token as strings. See internal ProSeries wiki for details.

#### (Optional) Developer Setup

For contributing to the `crash_analysis` library, it is recommended that you install extra python packages. 
Activate your cat python environment (step 4 in Set Up) and from the `CrashAnalysisTool` directory, run the following commands: 

```bash
pip install -r crash_analysis/module_requirements.txt
pip install -r crash_analysis/dev_requirements.txt
```

- `module_requirements.txt` include packages such as sci-kit learn and gensim, which are necessary for the machine-learning modules in the library (not currently publicly accessible). 
- `dev_requirements.txt` include packages that promote higher code quality, namely a python linter (flake8/hacking) and type checker (mylang).  

## Creating crash reports

1. Open a command prompt or terminal inside the `CrashAnalysisTool` directory on your machine. 

2. In the command prompt or terminal, start the jupyter notebook: `jupyter notebook`

3. A browser window should open up. Open `src/ExampleNotebook.ipynb`, **copy it** (File > Make A Copy...), and begin crash reporting!

### References

1. <a name="#1">[Learn X in Y Minutes where X = Python 3](https://learnxinyminutes.com/docs/python3/)</sup></a>

2. <a name="#2">[Jupyter Notebook Quickstart](https://jupyter.readthedocs.io/en/latest/content-quickstart.html)</sup></a>

3. <a name="#3">[10 minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)</sup></a>

## Latest Changes

#### Oct 26, 2017
- Revised documentation (this readme, docstrings in lib, and explicit comments in the example notebook)
- Added types and doctests to a few modules. 
- Added dev requirements

#### April 14, 2017
- Added quickbase downloader that can download crashed by time range in parallel
- Curated ExampleNotebook and CrashAnalysisTour
- Completely upgraded to Python 3
- (Finally) started writing documentation



