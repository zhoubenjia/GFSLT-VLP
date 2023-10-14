## How to install the nlgeval package?

Install the Python dependencies, run:
```bash
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```

If you encounter the following error, it means you need to install Java:

```bash
FileNotFoundError: [Errno 2] No such file or directory: 'java'
```
For convenience, you can first download the java installation package from [here](https://pan.baidu.com/s/1cbt3XKhRvvxIwEwnmZGLWg?pwd=gxup), then unzip it and move it to the ```/home/username``` directory:
```bash
mv jdk-17.0.4 /home/username/
```
Grant read and write permissions if necessary:
```bash
sudo chmod -R 777 jdk-17.0.4
```
After that, you need to add the following lines to your ```~/.bashrc```:
```bash
export JAVA_HOME=/homedata/username/jdk-17.0.4
export PATH=$PATH:$JAVA_HOME/bin
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
export JRE_HOME=$JAVA_HOME/jre
```
Finally execute the following command:
```bash
source ~/.bashrc
```
After completing these steps, enjoy the convenience brought by nlgeval.