# lumbar-ai
智能腰椎关键点与类型
### 训练
```shell script
python train.py -c hr_m --cfg config/w48_640_adam_lr1e-3.yaml
```
* 注：如果需要更换训练或验证数据集路径，请更改train.py的172-173行
```python
train_data_dir = os.getcwd()+r'/../data/train'
valid_data_dir = os.getcwd()+r'/../data/valid'
```


### 测试
```shell script
python test.py -c hr_m --cfg config/w48_640_adam_lr1e-3.yaml
```
* 注：如果需要更换测试数据集路径，请更改test.py的132行
```python
trainPath = os.getcwd()+r'/../data/test'
```


### 目录结构
```
|--project 
    |-- README.md
    |-- data
        |-- train
            |-- annotation.json
            |-- study*
        |-- valid
            |-- annotation.json
            |-- study*
        |-- test
            |-- study*
            |-- series_map.json
    |-- code
        |-- config
        |-- data
        |-- exp
        |-- models
        |-- task
        |-- utils
        |-- train.py
        |-- test.py
        |-- show.py
    |-- main.py or main.ipynb
    |-- submit
        |-- submit_20200203_040506.csv
```
