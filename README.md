# Virtual-Encoder-ECNDT-2023

This repository contains scripts which generate the results presented in ["Virtual encoder: a two-dimension visual odometer for NDT"](https://www.ndt.net/search/docs.php3?id=28119).

# 0. Table of Contents
1. [Demo video](#demo_video)
2. [Dataset](#dataset)
    1. [Dataset organization](#dataset_organization)
3. [Getting_started](#getting_started)
    1. [Installation](#getting_started_installation)
4. [Dependecies](#dependencies)

# 1. Demo video <a name="demo_video"></a>

[![](https://imagizer.imageshack.com/v2/640x480q70/922/k25IkO.png)](https://youtu.be/36NNLRFJXkg)
 
# 2. Dataset <a name="dataset"></a>

The dataset required to properly reproduce the results is avaliable at <a href="https://drive.google.com/drive/folders/1IEDjHuvrFpMRdltmPRAk1uatcMgQzyiS?usp=share_link"> https://drive.google.com/drive/folders/1IEDjHuvrFpMRdltmPRAk1uatcMgQzyiS?usp=share_link </a>. After downloading the "data.zip" you must copy and paste to the main project directory ("Virtual-Encoder-ECNDT-2023/") and then extract it.

## 2.1. Dataset Organization <a name="dataset_organization"></a>
Each folder represents an experiment (e.g. "air_cylindrical_longest_side") which contains:
<ol>
  <li>Photos shot by Raspberry Pi ground-facing camera;</li>
  <li>Orientation of the rigid body acquired by the inertial unit in quaternion and euler angle format;</li>
</ol> 
The experiments under "calibration" folder were used exclusively for calibration purpose. Data under "planar" and "cylindrical" folder were used for generating the paper results.

```
+---Virtual-Encoder-ECNDT-2023
|   |   data.zip
|   |   
|   \---data
|       +---calibration
|       |   +---air_cylindrical_longest_side
|       |   |       
|       |   +---air_cylindrical_shortest_side
|       |   |       
|       |   +---air_planar_longest_side
|       |   |       
|       |   +---air_planar_shortest_side
|       |   |       
|       |   +---water_cylindrical_longest_side
|       |   |       
|       |   +---water_cylindrical_shortest_side
|       |   |       
|       |   +---water_planar_longest_side
|       |   |       
|       |   \---water_planar_shortest_side
|       |   	| 
|       |           
|       +---cylindrical
|       |   +---air_closed_loop
|       |   |       
|       |   +---air_single_x
|       |   |       
|       |   +---air_single_y
|       |   |       
|       |   +---water_closed_loop
|       |   |       
|       |   +---water_single_x
|       |   |       
|       |   \---water_single_y
|       |           
|       \---planar
|           +---air_closed_loop
|           |       
|           +---air_single_x
|           |       
|           +---air_single_y
|           |       
|           +---water_closed_loop
|           |       
|           +---water_single_x
|           |       
|           \---water_single_y
|               
```

# 3. Dependecies <a name="dependencies"></a>
```
matplotlib # for data visualization
scipy # for signal processing algorithms
numpy # for general numerical processing
Pillow # for image reading and simple image manipulation (called PIL when imported)
```

# 4.Getting Started <a name="getting_started"></a>
## 4.1. Installation <a name="getting_started_installation"></a>
### 4.1.a Clone github repository
```
cd ~
git clone https://github.com/thiagokalid/Virtual-Encoder-ECNDT-2023
cd Virtual-Encoder-ECNDT-2023
```

### 4.1.b Prerequisities
You will need to build the Virtual-Encoder-ECNDT-2023 environment by following command:

