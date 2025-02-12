# CS499 Server Branch

## Running Everything

You can run (almost) everything easily on your own machine! The emulator has the exact same functionality as the Raspberry PI.
It sends data and connects to the server exactly the same, this is useful if you want to test stuff without having the PI.

To run the server on your machine use:
> python main.py

**Arguments**

`-e or --emulator`: Runs the emulator on your machine.

`-s or --stitch`: If we want to run the stitching function or not.

`-v [video] or --video [video]`: Chooses what video will be displayed on the emulator, you can pass 1 or 2 video names.

*Make sure that you are in the root directory to run `main.py` or if you aren't then make sure to pass the correct path to the file.*

***The Emulator functions exactly the same as the Raspberry PI. It sends the stream and connects to the socket the exact same way. Functional wise, they are the exact that the IP if the emulator is the same as the server, which is not important at all.***
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