# Single-Psychometric Function Goodness Of Fit: a Consistency measure
Use a package manager like anaconda to create an environment to work in. You do not want to install this into your base environment.

	conda create --name "testEnv"
	conda activate testEnv

install  this package to current environment with editing enabled

	(testEnv)$ cd sipgof
	(testEnv)$ pip install -e .

have a look at some examples

	(testEnv)$ python qualityMeasure/examples/LoadOMATestAndFit.py
	(testEnv)$ python qualityMeasure/adaptiveProcedureReader.py

it should fire up a plot and a rating

> Written with [StackEdit](https://stackedit.io/).
