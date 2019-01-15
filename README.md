# pytorch-playground
This repository contains sample projects developed with pytorch in order to get familiar with the stack.

## requirements

* Python 3.7.2
* Pytorch 1.0.0
* Matplotlib 3.0.1

## Known Bugs

1. On macOS *10.14.2* Matplotlib may runs into an **[NSApplication _setup:]: unrecognized selector sent to instance 0x7ffcf299e0b0** error. To avoid this issue inser the following lines to the top of the file.

    ```python
    from sys import platform as sys_pf
    if sys_pf == 'darwin':
        import matplotlib
        matplotlib.use("TkAgg")
    ```

    [Issue-Link](https://github.com/MTG/sms-tools/issues/36)

2. s