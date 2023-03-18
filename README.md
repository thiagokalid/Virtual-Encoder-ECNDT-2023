# Virtual-Encoder-ECNDT-2023

This repository contains scripts that generated the results presented in "Virtual encoder: a two-dimension visual odometer for NDT".

## 1. Dependecies
```
git status
git add
git commit
```

# 2. Dataset

The dataset required to properly reproduce the results are avaliable at <a href="https://drive.google.com/drive/folders/1IEDjHuvrFpMRdltmPRAk1uatcMgQzyiS?usp=share_link"> https://drive.google.com/drive/folders/1IEDjHuvrFpMRdltmPRAk1uatcMgQzyiS?usp=share_link </a> 

## 2.1. Dataset Organization
Each folder represents an experiment (e.g. "air_cylindrical_longest_side") which contains:
<ol>
  <li>Photos shot by the ground-facing camera;</li>
  <li>Orientation of the rigid body acquired by the inertial unit in quaternion and euler angle format;</li>
</ol> 
The experiments under "calibration" folder were used exclusively for calibration purpose. Data under "planar" and "cylindrical" folder was used for generating the paper results.

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

# 3.Getting Started
## 3.1. Instalation
### 3.1.a Clone github repository
```
cd ~
git clone https://github.com/thiagokalid/Virtual-Encoder-ECNDT-2023
cd Virtual-Encoder-ECNDT-2023
```

### 3.1.b Prerequisities
You will need to build the Virtual-Encoder-ECNDT-2023 environment by following command:

