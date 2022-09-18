# Zero-DCE

## Requirements

1. python 3.7.5
2. minspore-gpu 1.3.0
3. cuda 10.1

You'd better install the mindspore-gpu follow the [official instruction](https://www.mindspore.cn/install).

or you can create a conda environment to run our code like this: 
```bash
$ conda env create -f ./mindspore1.3.yaml
```

## Train

```bash
$ python ./lowlight_train.py
```
Use `python ./lowlight_train.py --help` for more details.

## Test

```bash
$ python ./lowlight_test.py
```

Use `python ./lowlight_test.py --help` for more details.

## Acknowledge

Acknowledgements to [the mindspore team](https://www.mindspore.cn/) who maintain the mindspore including the [modle zoo](https://gitee.com/mindspore/models).
