# CS499 Server Branch

## Running the Program

To run the program all you need to do is type the following in the terminal:

>
> python main.py
> 
Make sure that you have installed all the dependencies in `requirements.txt`.

## Configuring the Program

The configuration system is constantly being worked on but here is what is has currently. Here is `config.json` where 
all the configurations are stored.

```json
"emulator_arguments": {
  "enabled": boolean,
  "stream_enabled": boolean,
  "video_name": string
}
```
This is for the emulator.`enabled` dictates if the emulator is enabled. 
`stream_enabled` dictates if there will be a camera feed opened for the emulator stream. If `false`, then the video will be used instead.
`video_name` is the name of the video used under `assets/videos`.

```json
"server_port": int
```
This is the port the server will open on for the camera client to connect to.

```json
"vision_arguments": {
    "enabled": true,
    "depth_threshold": 0.8,
    "StereoSGBM_args": {
      "minDisparity": 0,
      "numDisparities": 32,
      "blockSize": 5,
      "uniquenessRatio": 30,
      "speckleWindowSize": 100,
      "speckleRange": 2,
      "disp12MaxDiff": 1
    },
    "scale": 0.5,
    "calibration_file": "calib_50/calibration_50.npz",
    "camera_parameters": {
      "baseline": 0.07,
      "viewing_angle": 120
    }
  }
```
This is for the computations done on the frames in the frame queue. `StereoSGB_args` determines the parameters for the StereoSGBM_args for the algorithm.
`depth_threshold` is the threshold for the depth algorithm on detecting how close things are. `scale` determines the scale of the image size and resolution for the algorithm.
And `calibration_file` is the NPZ calibration file used for the setup.

```json
"camera_parameters": {
    "baseline": float,
    "viewing_angle": float
  }
```
This is for the physical camera values that will be needed for calibration and depth estimation. These are purely its physical values, so you should not change this configuration unless you are using an emulated camera that you need to calibrate.
`baseline` is the distance between the sensors. `focal_length` si the focal length of the camera, since there are 2 identical ones we only need to store one value.
`viewing_angle` is the FOV of the cameras, currently it is locked to 120 degrees.