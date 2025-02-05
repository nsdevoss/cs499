# CS499 Server Branch

## Running Everything

You can run (almost) everything easily on your own machine! The emulator has the exact same functionality as the Raspberry PI.
It sends data and connects to the server exactly the same, this is useful if you want to test stuff without having the PI.

To run the server on your machine use:
> python main.py

To run the server and the emulator on your machine use:
> python main.py -e
> 
> or
> 
> python main.py --emulator

To change the video the emulator plays use:
> python main.py -e -v *video_name*
> 
> or
> 
> python main.py -e --video *video_name*

Make sure that you are in the root directory to run `main.py` or if you aren't then make sure to pass the correct path to it.

## Running parts individually

### Running the server
To run the server individually you need to run src/server/server.py.
This will create a server on your machine listening on port 9000. I'm working on making the port configurable.

### Running the Raspberry PI
To run on the Raspberry PI individually, you need to run src/pi/client_pi.py on the Raspberry PI.
The PI must be on the same Wi-Fi as the server in order to connect to it.

### Running the Emulator
To run the emulator individually, you need to run src/pi/emulator.py on your machine.
This will simulate the Raspberry PI and will be exactly the same in sending data and connection.