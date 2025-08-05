# Single-Psychometric Function Goodness Of Fit: a Consistency measure
Use a package manager like anaconda to create an environment to work in. You do not want to install this into your base environment.

	conda create --name "testEnv"
	conda activate testEnv
	conda install pip

install this package to current environment with editing enabled. 
The promptline should change to something like this: **(testEnv) me@myPc ~/pathToSiPGOF$**

	cd sipgof
	pip install -e .

## have a look at this example:

	python qualityMeasure/examples/LoadOMATestAndFit.py
 
should fire up a plot and a rating.

> Written with [StackEdit](https://stackedit.io/).
