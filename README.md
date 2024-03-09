# Digital-image-processing
Using conventional &amp; DL method to implement image segmentation in small dataset.
Conventional Method

1.程式碼中第14行，改成待測資料的位置
2.程式碼中第88行，改成結果的圖像名字
3.程式碼中第89行，改成結果的圖像位置
4.確定以上都沒錯便可執行

Deep learning method

Require environment
# Required dependencies ------------------------------------------------------------------------------------------------
dependencies = [
    "matplotlib>=3.3.0",
    "numpy>=1.22.2",
    "opencv-python>=4.6.0",
    "pillow>=7.1.2",
    "pyyaml>=5.3.1",
    "requests>=2.23.0",
    "scipy>=1.4.1",
    "torch>=1.8.0",
    "torchvision>=0.9.0",
    "tqdm>=4.64.0", # progress bars
    "psutil", # system utilization
    "py-cpuinfo", # display CPU info
    "thop>=0.1.1", # FLOPs computation
    "pandas>=1.1.4",
    "seaborn>=0.11.0", # plotting
]

pip install the ultralytics package including all requirements in a Python>=3.8 environment with PyTorch>=1.8.

If you don't have the package ultralytics
type
<pip install ultralytics> in the command prompt

We only use the validation part of YOLOv8
Make sure to check out the directory you at.
Start the validation process and you will get two folder "Detect_origin_DL" & "Detect_mask_DL"
