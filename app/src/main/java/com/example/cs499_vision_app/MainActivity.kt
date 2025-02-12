package com.example.cs499_vision_app
import android.content.Context
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.cs499_vision_app.ui.theme.Cs499_vision_appTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {

            MainScreen()

        }
    }
}
// Main screen settings
@Composable
fun MainScreen(){
    Scaffold { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally

        ) {
            VibrateButton()
        }
    }
}
//function to call triggerVibration for a button
@Composable
fun VibrateButton() {
    val context = LocalContext.current

    Button(
        onClick = { triggerVibration(context) },
        modifier = Modifier.padding(16.dp)
    ) {
        Text("Vibrate")
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