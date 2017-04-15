# ProSeries Crash Analysis Tool (CAT)

CAT is a tool for offline crash report analysis. CAT allows faster, more precise queries of ProSeries crash data.  It includes a quickbase crash report downloader, xml parser, Pandas dataframe helper functions, and some text analysis tools.

If you are new to Python 3[<sup>1</sup>](#references), Jupyter notebooks[<sup>2</sup>](#references), or Pandas[<sup>3</sup>](#references), check out the references section.

After installation, check out `CrashAnalysisTour.ipynb` to see examples of the commands that are available with CAT. Copy and rename `ExampleNotebook.ipynb`
to create a new crash report.

## Installation

### Prerequisite Software

Python 3 is required (3.5+ preferred). We recommend installing python with [Anaconda](https://www.continuum.io/downloads).

[PyCharm](http://jetbrains.com/pycharm) or the [Visual Studio Python Plugin](https://github.com/Microsoft/PTVS/) is recommended, but not required.

Git is also required.

### Set Up
In a terminal or command prompt, do the following:

1. Download this repository: `git clone https://github.intuit.com/arosengarten/CrashAnalysisTool.git`

2. Go inside the directory: `cd CrashAnalysis`

3. Create a virtual environment: `conda create --name cat35 python=3.5`

4. Activate the virtual environment (of python 3.5): `source activate cat35` for OSX/Linux, or `activate cat35` for Windows.

5. Install required python packages: `pip install -r requirements.txt`

6. Open `crash_analysis/private.py` and input the database id, username, password, and app token as strings. See internal ProSeries wiki for details.

7. Start the jupyter notebook: `jupyter notebook`


### References

1. <a name="#1">[Learn X in Y Minutes where X = Python 3](https://learnxinyminutes.com/docs/python3/)</sup></a>

2. <a name="#2">[Jupyter Notebook Quickstart](https://jupyter.readthedocs.io/en/latest/content-quickstart.html)</sup></a>

3. <a name="#3">[10 minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)</sup></a>

## Latest Changes

#### April 14, 2017
- Added quickbase downloader that can download crashed by time range in parallel
- Curated ExampleNotebook and CrashAnalysisTour
- Completely upgraded to Python 3
- (Finally) started writing documentation



