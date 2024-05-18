# Deep linear networks for regression are implicitly regularized towards flat minima

## Environment

### With conda

```
conda env create -f environment.yml
```

### With pip

Install Python 3.9.19 and pip 24.0, then

```
pip3 install -r requirements.txt
```

## Reproducing the experiments of the paper

Run

```
python run_experiment.py
```

The code takes around 40 minutes to run on a laptop CPU.