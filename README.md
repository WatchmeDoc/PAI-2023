<div align="center">

![Alt](assets/eth_logo.png "Title")
# Probabilistic Artificial Intelligence
### Fall Semester 2023-2024
### Professor: [Andreas Krause](https://las.inf.ethz.ch/krausea)
    

<a href="#">
    <img src="https://img.shields.io/badge/Python-3.8, 3.9, 3.10-1cb855">
</a>
<a href="#">
    <img src="https://img.shields.io/badge/Docker-4.23-0388fc">
</a>
<a href="#">
    <img src="https://img.shields.io/badge/License-MIT-8a0023">
</a>
<br>
<a href="https://las.inf.ethz.ch/teaching/pai-f23"><strong>Explore Course Page »</strong></a>
</div>

## Project Description
Project for Probabilistic AI (PAI) course at ETH Zürich.


## Getting Started
The project is tested on docker. Make sure you have `Docker` and `docker-compose` installed on your machine.
To simply run the project without the need of docker, you can use `conda` to create a virtual environment.
### Running it locally
First, create a conda environment. The recommended python version is `3.8`.
```sh
$ conda create -n PAI python=3.8 -y && conda activate PAI
```
Then, install all required packages:
```shell
$ pip install -r <filepath>/requirements.txt
```
Now you can run the corresponding project file you want (i.e. `task00/task0_handout_5slq29/solution.py`)
### Testing
To test the project, there is a script file `runner.sh` that runs all the project files and saves the results in `results/` folder.
On Linux and MacOS, simply run:
```shell
$ bash runner.sh
```
On Windows, you can use `git bash` or `WSL` to run the script. Otherwise, open a Powershell, change the directory to the
handout folder and run:
```shell
$ docker build --tag task0 .; docker run --rm -v "$(pwd):/results" task0
```

#### Team Members:
* Aris Koutris, [akoutris@ethz.ch](mailto:akoutris@ethz.ch)
* Aylin Akkus,  [aakkus@ethz.ch](mailto:aakkus@ethz.ch)
* George Manos, [gmanos@ethz.ch](mailto:gmanos@ethz.ch)
