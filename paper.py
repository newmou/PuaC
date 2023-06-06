import paperPy
import numpy as np

# 生成输入数组
fitmols = np.random.rand(numFitMols, numFitAtoms, numTransforms).astype(np.float32)
refmol = np.random.rand(numAtoms, 4).astype(np.float32)

# 调用函数
gpuID = 0
timingActivated = True
result = your_module_name.optimize_sepkernels_py(fitmols, refmol, gpuID, timingActivated)

# 打印结果
print(result)
