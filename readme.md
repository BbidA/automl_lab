# 实验代码执行说明

## 环境说明

之前实验中所使用的环境如下

### Linux

kernel : 3.10.0-862.9.1.el7.x86_64

```
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"

CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"
```

### Python 

（各个包的新的版本应该也可以，但是可能有版本兼容问题）

1. python 3.6.7
2. scikit-learn 0.20.1
3. scipy 1.1.0
4. pandas 0.23.4
5. numpy 1.15.4

## 运行

### 运行命令

```shell
export PYTHONPATH="$PYTHONPATH:$(pwd)"  # 首先需要把 automl_lab 目录加到 python 的运行环境里面去
cd automl_lab/bandit
python3 test.py ground  # find ground truth model
python3 test.py random  # random search
python3 test.py sf      # softmax	
python3 test.py eg      # epsilon-greedy
python3 test.py ucb		# ucb
python3 test.py proposed theta gamma beta  # proposed method
```

### 数据集设置

数据集是通过之前整理好的 data_set 中的 pickle 文件加载之后得到的，如果要添加或删除数据集，在 `temp_dataset` 文件夹中添加或删除相应的 `*_train_data.pkl` 和 `*_test_data.pkl` 文件即可（train data 和 test data 必须同时添加或删除，否则程序会出错）

### 参数设置

- budget 设置：打开 test.py 文件，修改 BUDGET 变量的值即可

- epsilon-greedy 的 epsilon：修改 `test.py` 文件中的 `eg_method` 方法中下面一行代码参数即可

  ```python
  best_optimization = model_selection.fit(train_x, train_y, budget=BUDGET, epsilon=0.1)
  ```

- softmax 的 temperature：修改 `test.py` 文件中的 `softmax_method` 方法中下面一行代码的参数

  ```python
  best_optimization = model_selection.fit(train_x, train_y, temperature=0.5, budget=BUDGET)
  ```

- proposed 的三个参数：按照命令行依次输入参数即可

## 结果说明

在 `automl_lab/bandit/log` 文件夹中

1. `*.log` 文件均记录了程序运行的中间信息（运行时间，运行到哪一步等）
2. 各个选择方法对应的文件夹中的 `methodname_datasetname.csv` 文件记录了每个数据集所有模型的均值、标准差、budget 和 best v
3. 各个选择方法对应的文件夹中的 `methodname_lab.csv` 是对结果的汇总
4. `gt_dataset.csv` 记录了该数据集上每个 model 评估值的均值、最大值和标准差

