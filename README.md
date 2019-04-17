# Deep RL Quadcopter Controller

*Teach a Quadcopter How to Fly!*

In this project, you will design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm of your choice! 

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.

```
git clone https://github.com/tobiassteidle/MLND_Quadcopter.git
cd MLND_Quadcopter
```

2. Create and activate a new environment.

```
conda create -n quadcop python=3.6 matplotlib numpy pandas
source activate quadcop
```

3. Install Dependencies.
```
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

4. Tests Tensorflow GPU (optional)
```
python test_tensorflow_gpu.py
```

Should output something like this:
```
2019-04-16 12:06:23.778543: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX
2019-04-16 12:06:24.060124: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1105] Found device 0 with properties:
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.7465
pciBusID: 0000:02:00.0
totalMemory: 8.00GiB freeMemory: 6.64GiB
2019-04-16 12:06:24.067366: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1195] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1070, pci bus id: 0000:02:00.0, compute capability: 6.1)
2019-04-16 12:09:05.498872: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1195] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1070, pci bus id: 0000:02:00.0, compute capability: 6.1)
Default GPU Device: /device:GPU:0
```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `quadcop` environment. 
```
python -m ipykernel install --user --name quadcop --display-name "quadcop"
```

6. Open the notebook.
```
jupyter notebook Quadcopter_Project.ipynb
```

7. Before running code, change the kernel to match the `quadcop` environment by using the drop-down menu (**Kernel > Change kernel > quadcop**). Then, follow the instructions in the notebook.
