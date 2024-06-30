# Tennis Ball Trajectory Computation using Stereo Vision

This project uses stereo vision with two cameras to track a tennis ball on a 3D court.

## Process Overview

1. **Ball Detection:** The ball is detected and its coordinates are extracted in the two frames via color filtering.
2. **3D Coordinates Calculation:** Using photogrammetry formulas and the coordinates from both the left and right image frames, the 3D coordinates of the ball within each camera's reference frame are calculated.
3. **Common Reference System Transformation:** The 3D coordinates are transformed into a common reference system to visualize the ball's trajectory on the 3D court.

The footage comes from a tennis simulation developed in Unity, designed to capture the scene simultaneously from two different camera perspectives.

## Example of Result
![Trajectory in 3D coordinate system](https://github.com/rantaluca/ball_tracking/assets/102813576/f1f78b13-3b47-49b2-bedc-623e04f927a1)

## Algorithm Formulas

The algorithm to compute the 3D position from the stereo images uses the following photogrammetry formulas:

Given:

- $`e`$ : distance between the two cameras
- $`f `$: focal length
- $`x_R `$: x-coordinate in the right camera image
- $`x_L `$: x-coordinate in the left camera image
- $`y_R `$: y-coordinate in the right camera image
- $`y_L `$: y-coordinate in the left camera image

### Disparity Calculation

The disparity $` d `$ is calculated as:
$` d = x_R - x_L `$


### 3D Coordinates Calculation

The 3D coordinates of the ball in one of the camera's reference frame are calculated as:

$`
 x = \frac{x_R \cdot e}{d} 
 y = \frac{y_R \cdot e}{d} 
 z = \frac{e \cdot f}{d} 
`$

### Transformation to the terrain reference frame  

The transformation matrix $` T `$ is applied to convert coordinates to the terrain reference frame:

$` T = \begin{bmatrix}
0.96592583 & 0 & 0.25881905 & -9.3 \\
0 & 1 & 0 & -3.5 \\
-0.25881905 & 0 & 0.96592583 & -8.2 \\
0 & 0 & 0 & 1
\end{bmatrix} `$

The transformed coordinates  $`\mathbf{P}_{transformed}`$  are obtained by:

$` \mathbf{P}_{transformed} = T \cdot \mathbf{P} `$

where ($` \mathbf{P} `$)s the homogeneous coordinate vector of the 3D points.

## Video from the Left Before Processing 
[![Left Video](https://github.com/rantaluca/ball_tracking/assets/102813576/69435848-0b3f-45fc-94bc-c70752da114d)](https://github.com/rantaluca/ball_tracking/assets/102813576/69435848-0b3f-45fc-94bc-c70752da114d)

## Mask with the Ball Color
![Mask](https://github.com/rantaluca/ball_tracking/assets/102813576/bb6f24d6-d586-4a89-89e0-c222867ac8fa)

## Contour Detection to Extract the Ball Coordinates on the Image
![Contours](https://github.com/rantaluca/ball_tracking/assets/102813576/669b6237-f1ab-45ca-865b-d2f8cfe60376)

## Result
![Result](https://github.com/rantaluca/ball_tracking/assets/102813576/21a5eee0-e6c7-4fe9-8147-3ea805cee3f6)


## How to Run the Code

### Step 1: Install the required libraries

In your terminal, run:
```sh
pip3 install opencv-python
pip3 install numpy
pip3 install matplotlib
```
### Step 2: Clone the repository:

```sh
git clone [https://github.com/rantaluca/ball_tracking.git](https://github.com/rantaluca/ball_tracking.git)
cd ball_tracking
```

### Step 3: Add your video files (They should be identical from two different camera frames):
    - Place your left camera video in the `videos` directory and name it `left_short.mp4`.
    - Place your right camera video in the `videos` directory and name it `right_short.mp4`.

### Step 4: Run the script:
**Note**: The transformation matrix  T  will need to be redefined depending on the positions of the cameras relative to the tennis court in your scene. Check this link to learn about the transformation matrix.

![transformation matrix ](https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html)

```sh
    python ball_tracking_clean.py
```
### Step 5: Run the script: View the output:
    - The processed videos and images will be saved in the `videos` and the current directory respectively.
    - The 3D trajectory plot will be displayed and saved as `Trajectoire_3D_repere.png`.

**Note**: A version with a one-line command to define the two source videos and only display the result is coming soon.
