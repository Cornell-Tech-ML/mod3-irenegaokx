# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## 3.2: Parallel_check output
[View output here](output_files/parallel.txt)

## 3.4: GPU and CPU comparison
Timing summary
Size: 64
    fast: 0.00307
    gpu: 0.00610
Size: 128
    fast: 0.01520
    gpu: 0.01332
Size: 256
    fast: 0.10017
    gpu: 0.04800
Size: 512
    fast: 0.98971
    gpu: 0.20878
Size: 1024
    fast: 10.55691
    gpu: 0.88907

![GPU vs CPU time](output_files/output.png)


## 3.5
## CPU, Hidden 100
Split:
[Split: View output here](output_files/split_cpu_100.txt)

