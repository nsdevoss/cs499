package com.example.cs499_vision_app
import android.content.Context
import android.media.MediaPlayer
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.selection.selectable
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp

import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.core.view.ViewCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.PrintWriter
import java.net.Socket
import kotlinx.coroutines.CompletableDeferred
import androidx.core.view.WindowInsetsCompat

import androidx.compose.ui.graphics.Color
import androidx.compose.material3.lightColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.res.painterResource
import androidx.compose.foundation.Image

import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.sp

import com.example.cs499_vision_app.R

import com.example.cs499_vision_app.ui.theme.Cs499_vision_appTheme

class MainActivity : ComponentActivity() {


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {

            val customColorScheme = lightColorScheme(
                primary = Color(0xFF62AB37),        //earthy green
                onPrimary = Color.White,
                secondary = Color(0xFF345511),      //deep forest green
                onSecondary = Color.White,
                background = Color(0xFFEADACB),     //soft rosy beige
                onBackground = Color(0xFF393424),   // Complementary (for text on background)
                surface = Color.White,
                onSurface = Color(0xFF393424)       // Also using complementary for contrast

            )

            val Quicksand = FontFamily(
                Font(R.font.quicksand_regular, FontWeight.Normal),
                Font(R.font.quicksand_bold, FontWeight.Bold)
            )

            val CustomTypography = Typography(
                bodyLarge = TextStyle(
                    fontFamily = Quicksand,
                    fontWeight = FontWeight.Normal,
                    fontSize = 16.sp
                ),
                bodyMedium = TextStyle(
                    fontFamily = Quicksand,
                    fontWeight = FontWeight.Normal,
                    fontSize = 14.sp
                ),
                titleLarge = TextStyle(
                    fontFamily = Quicksand,
                    fontWeight = FontWeight.Bold,
                    fontSize = 22.sp
                ),
                headlineMedium = TextStyle(
                    fontFamily = Quicksand,
                    fontWeight = FontWeight.Bold,
                    fontSize = 28.sp
                )
            )

            MaterialTheme(colorScheme = customColorScheme,
            typography = CustomTypography
            )
            {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                    contentColor = MaterialTheme.colorScheme.onBackground
                )
                {

                    MainScreen()
                }
            }

        }
    }
}

private lateinit var socket: Socket
private lateinit var brInput: BufferedReader
private lateinit var brOutput: PrintWriter
private val connectionReady = CompletableDeferred<Unit>()


// Main screen settings
@Composable
fun MainScreen(){
    // automatically select just vibrate
    var selectedOption by remember {mutableStateOf("Vibrate")}
    var serverMessage by remember { mutableStateOf("no message yet")}
    val context = LocalContext.current

    Scaffold(
        topBar = { AppHeader()},
    content = {
        innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally

        ) {



            SelectionMenu(selectedOption) {selectedOption = it}

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = {
                    when(selectedOption) {
                        "Vibrate" -> triggerVibration(context)
                        "Sound" -> playSound(context)
                        "Both" -> {
                            triggerVibration(context)
                            playSound(context)
                        }
                    }
                }
            ) {
                Text("Play")
            }

            Spacer(modifier = Modifier.height(16.dp))
            Button(onClick = {
                connect(context){
                    msg->serverMessage =msg
                }
            }) {
                Text("Connect")
            }
            Spacer(modifier = Modifier.height(32.dp))
            Text("Server message: $serverMessage")
        }


    }
            )
}

@Composable
fun AppHeader(){
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ){
        Text(text="Oculosarus",
            style =MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.secondary)

        Image(
            painter = painterResource(id=R.drawable.oculosarus_logo),
            contentDescription = "Oculosarus Logo",
            modifier = Modifier
                .size(100.dp)
        )
    }
}

// making a radiobutton option for sound and vibration preference
//https://developer.android.com/develop/ui/compose/components/radio-button
@Composable
fun SelectionMenu(selectedOption: String, onOptionSelected: (String) -> Unit) {
    val radioOptions = listOf("Vibrate", "Sound", "Both")

    Column {
        radioOptions.forEach {text->
            Row(
                Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .selectable(
                        selected = (text == selectedOption),
                        onClick = {onOptionSelected(text)},

                    )
                    .padding(horizontal = 16.dp),
                verticalAlignment = Alignment.CenterVertically
                    )
            { RadioButton(
                selected = (text == selectedOption),
                onClick = null

            )
            Text(text = text,
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(start = 16.dp))
            }}

        }
    }



// function to trigger a vibration
fun triggerVibration(context: Context) {
    val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        vibratorManager.defaultVibrator
    } else {
        context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
    }


    if (vibrator.hasVibrator()) {
        val effect = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            // vibrate one time for 500 ms
            VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE)
        } else {
            null
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(effect!!)
        } else {
            vibrator.vibrate(500) // Deprecated for newer versions
        }
        // Toast message when vibrate is called
        Toast.makeText(context,"Device Vibrated", Toast.LENGTH_SHORT).show()

    }
    else{
        Toast.makeText(context, " Device did not vibrate", Toast.LENGTH_SHORT).show()
    }
}

fun playSound(context: Context){
    // sound file located in app/src/main/res/raw/
    val mediaPlayer = MediaPlayer.create(context, R.raw.mixkit_magic_marimba_2820) // sound file https://mixkit.co/free-sound-effects/notification/
    mediaPlayer.start()
    Toast.makeText(context, "Playing Sound", Toast.LENGTH_SHORT).show()
}

fun connect(context: Context,
            onMessageReceived:(String) -> Unit

) {
    Log.d("Connection attempt", "172.24.24.166 && 12345")
    CoroutineScope(Dispatchers.Main).launch {
        val txtFromServer = withContext(Dispatchers.IO) {

            socket = Socket("172.24.24.166", 12345)
            brInput = BufferedReader(InputStreamReader(socket.getInputStream()))
            brOutput = PrintWriter(socket.getOutputStream())

            brOutput.write("hello from app\n")
            brOutput.flush()
            brInput.readLine()

        }
        connectionReady.complete(Unit)
        Toast.makeText(context, "Server message: $txtFromServer", Toast.LENGTH_LONG).show()
        onMessageReceived(txtFromServer)

        readMessagesInBackground(onMessageReceived)
    }
}

fun readMessagesInBackground(onMessageReceived: (String) -> Unit) {
    CoroutineScope(Dispatchers.IO).launch {
        try {
            connectionReady.await()
            while(true) {
                val msg = brInput.readLine() ?: break
                Log.d("Server Message" , msg)
                withContext(Dispatchers.Main) {
                    onMessageReceived(msg)
                }
            }
        }catch (e:Exception) {
            Log.e("Read Error", e.message ?: "Unknown Error")
        }
    }
}
