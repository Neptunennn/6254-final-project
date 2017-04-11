# How To Use Virtualenv In ECE6254 Project
## Install virtualenv:
```bash
$ pip install virtualenv
$ pip install virtualenvwrapper
$ export WORKON_HOME=~/Envs
$ source /usr/local/bin/virtualenvwrapper.sh
```
(optional)
add
```bash
export WORKON_HOME=~/Envs
source /usr/local/bin/virtualenvwrapper.sh
```
to your bashrc file. I don't know how to do this in Mac, so you can either search the internet or execute these commands every time you open your terminal.

If you want to do the instructions above on your **Mac**, you have to find the path of ```virtualenvwrapper.sh```. If you have already installed the Anaconda, it may be under the ```/Users/{your account name here}/anaconda/bin/```.
What you have to do is edit your ```~/.bash_profile``` file. 

However, I faced some problems when make the virtual environment. I went through the issue report on their Github and found that it is a problem caused by the Anaconda.

## Basic Usage
1. Create a virtual environment:
```bash
$ mkvirtualenv -p /usr/bin/python2.7 ece6254
```
This creates the `ece6254` folder inside `~/Envs`. `-p /usr/bin/python2.7` means we will use python2.7.

2. Work on a virtual environment:
```bash
$ workon ece6254
```

3. Install requirements:
```bash
$ pip install -r requirements.txt
```

4. Deactivate the virtualenv:
```bash
$ deactivate
```

## Reference
[http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/](http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/)
