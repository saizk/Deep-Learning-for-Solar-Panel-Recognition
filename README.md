# Deep-Learning-for-Solar-Panel-Recognition

Recognition of photovoltaic cells in aerial images with **Convolutional Neural Networks** (CNNs).
**Object detection** with YOLOv5 models and **image segmentation** with Unet++, FPN, DLV3+ and PSPNet.


## ๐ฝ Installation + pytorch CUDA 11.3

-----------
With **pip**:
```
pip3 install -r requirements.txt && pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```
With **Anaconda**:
```
pip3 install -r requirements.txt && conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## ๐ Data gathering

-----------
* ### โ Solar Panels Dataset
    _Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery_ (https://zenodo.org/record/5171712)
  ![](reports/figures/sp_dataset.png)
* ### ๐ Google Maps Aerial Images
  * **GoogleMapsAPI:** ``src/data/wrappers.GoogleMapsAPIDownloader``
  * **Web Scraping:** ``src/data/wrappers.GoogleMapsWebDownloader``
  ![](reports/figures/gmaps.png)
* ### ๐ก Sentinel-2 Data (unused)
  Sentinel-2 Satellite data from Copernicus. ``src/data/wrappers.Sentinel2Downloader``

## ๐  Processing pipeline

------------
![pipeline](reports/figures/data_pipeline.png)

## ๐งช Models

-----------
* ### Object Detection
  * **YOLOv5-S:** 7.2 M parameters
  * **YOLOv5-M:** 21.2 M parameters
  * **YOLOv5-L:** 46.5 M parameters
  * **YOLOv5-X:** 86.7 M parameters

  Architectures are based on [YOLOv5](https://github.com/ultralytics/yolov5) repository.

* ### Image Segmentation
  * **Unet++:** ~ 20 M parameters
  * **FPN:** ~ 20 M parameters
  * **DeepLabV3+:** ~ 20 M parameters
  * **PSPNet:** ~ 20 M parameters

  Architectures are based on [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) repository.


## ๐ Results

---------------
* ### Metrics
![Object Detection vs Image Segmentation](reports/figures/od_vs_is.png)
* ### Dataset and Google Maps images
![Object Detection vs Image Segmentation](reports/figures/sp_results.png)


๐ Project Organization
------------

    โโโ LICENSE
    โโโ Makefile           <- Makefile with commands like `make data` or `make train`
    โโโ README.md          <- The top-level README for developers using this project.
    โโโ data               <- Data for the project (ommited)
    โโโ docs               <- A default Sphinx project; see sphinx-doc.org for details
    โ
    โโโ models             <- Trained and serialized models, model predictions, or model summaries
    โ
    โโโ notebooks          <- Jupyter notebooks.
    โ        โโโ pytorch_lightning.ipynb            <- Modeling with Pytorch Ligthning.
    โ        โโโ pytorch_sp_segmentation.ipynb      <- Modeling with vanilla Pytorch.
    โ
    โโโ references         <- Data dictionaries, manuals, and all other explanatory materials.
    โ
    โโโ reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    โ        โโโ figures        <- Generated graphics and figures to be used in reporting
    โ        โโโ Solar-Panels-Project-Report-UC3M         <- Main report
    โ        โโโ Solar-Panels-Presentation-UC3M.pdf       <- Presentation slides for the project.
    โ
    โโโ requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    โ                         generated with `pip freeze > requirements.txt`
    โ
    โโโ setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    โโโ src                <- Source code for use in this project.
    โ       โโโ __init__.py    <- Makes src a Python module
    โ       โ
    โ       โโโ data           <- Scripts to download or generate data
    โ       โ       โโโ download.py   <- Main scripts to download Google Maps and Sentinel-2 data. 
    โ       โ       โโโ wrappers.py   <- Wrappers for all Google Maps and Sentinel-2.
    โ       โ       โโโ utils.py      <- Utility functions for coordinates operations.
    โ       โ
    โ       โโโ features       <- Scripts to turn raw data into features for modeling
    โ       โ       โโโ create_yolo_annotations.py   <- Experimental script to create YOLO annotations.
    โ       โ       โโโ preprocess_data.py           <- Script to process YOLO annotations.
    โ       โ
    โ       โโโ models         <- Scripts to train models and then use trained models to make predictions
    โ       โ       โโโ segmentation  <- Image segmentation scripts to train Unet++, FPN, DLV3+ and PSPNet models.
    โ       โ       โโโ yolo          <- Object detection scripts to train YOLO models.
    โ       โ
    โ       โโโ visualization  <- Scripts to create exploratory and results oriented visualizations
    โ            โโโ visualize.py
    โ
    โโโ tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
--------
