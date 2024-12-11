# Enhancing Hand-Object Interaction Pose Reconstruction through Semantic-Enhanced and Reconstruction Modules


# Denpendencies
```python
manopth==0.0.1  <br>
matplotlib==3.3.4  <br> 
numpy==1.13.3  <br> 
opencv_python==4.5.3.56  <br> 
Pillow==9.2.0  <br> 
pymeshlab==2021.10  <br> 
scikit_image==0.17.2  <br> 
scipy==0.19.1  <br> 
skimage==0.0  <br> 
torch==1.10.1  <br> 
torchvision==0.11.2  <br> 
tqdm==4.62.3
```

# Create Dataset files
HOR2H_main/dataset/make_data.py creates the dataset files for HO3D by creating train-val-test splits and preprocess them to problem needs.<br>
HOR2H_main/innovation/make_dataset_h2o3d.py creates the dataset files for H2O3D by creating train-val-test splits and preprocess them to problem needs.<br>
Adapt the variables (root and mano_root) to the HO3D dataset path (click [here](https://www.tugraz.at/index.php?id=40231) to download HO3D) and mano models path (click [here](https://mano.is.tue.mpg.de/) to download MANO).
Adapt the variables (root and mano_root) to the H2O3D dataset path (click [here](https://www.tugraz.at/index.php?id=57823) to download HO3D) and mano models path.<br>
running the following command will create the dataset files and store them in ./datasets/ho3d/
```python
python3 datasets/make_data.py --root /path/to/HO3D --mano_root /path/to/MANO --dataset_path ./datasets/ho3d/
```
running the following command will create the dataset files and store them in ./datasets/h2o3d/
```python
python3 datasets/make_data.py --root /path/to/H2O3D --mano_root /path/to/MANO --dataset_path ./datasets/h2o3d/
```

# Train model
Run main_HOR2H.py to train the model from scratch. For parameter Settings, see .\scripts\train_ho3d.sh or .\scripts\train_h2o3d.sh. See utils/options.py for a detailed description of the parameters.<br>
running the following command to train the model
```python
python3 main_HOR2H.py
```

# Evaluate and Visualize model
Run test_HOR2H.py to evaluate the model or generate a visualization (set the parameter --visualize the visualization). For details about parameter Settings, see. \scripts\test_ho3d.sh or. \scripts\test_h2o3d.sh.<br>
Run the following command to evaluate or visualize
```python
python3 test_HOR2H.py
```
We are providing pretrained weights for models that were trained on datasets of hand and object interactions, so that users can test and see the results of the model without having to train from scratch:<br>
Click [here](https://pan.baidu.com/s/1XbE48bj5XcbFRuR-sALu-g?pwd=eur8) to download pretrained weights. Place pre-training weights under the appropriate paths in.\checkpoints.

# Acknowledgements
Our implementation is built on top of multiple open-source projects.  We would like to thank all the researchers who provided their code publicly:<br>
[THOR-Net](https://github.com/ATAboukhadra/THOR-Net)<br>
[GraFormer](https://github.com/Graformer/GraFormer)<br>
[SemGCN](https://github.com/garyzhao/SemGCN)<br>



