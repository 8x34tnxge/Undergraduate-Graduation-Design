# Resource Assignment Algorithm

## Usage

Before configure the necessary python environment and use this repo, you had better installed [make](https://github.com/mirror/make), [pyenv](https://github.com/pyenv/pyenv) & [pipenv](https://github.com/pypa/pipenv) respectively.

### Configure the environment

First you can install python via `pyenv` using the following command:

```bash
    pyenv install 3.7
```

Then use the following command to auto-configure the environment via `Pipfile` and `Pipfile.loc` files:

```bash
    pipenv install --dev
```

> Notice: If you don't want to install the log module `loguru`, you can ignore the arg `--dev`

### Run the demo

if you just wanna run the mode, just use this short command:

```bash
    make
```

or customize the makefile to reach your requirements. 