package com.andre.tflite.classification

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import com.andre.tflite.classification.databinding.ActivityMainBinding

class MainActivity : ComponentActivity() {

    private lateinit var binding: ActivityMainBinding

    private var selectedBitmap: Bitmap? = null
    private var segmenter: Segmenter? = null

    // Definindo imageSize e outputChannels para o modelo de segmentação
    private val IMAGE_SIZE = 256 // Tamanho da imagem de entrada para o modelo
    private val OUTPUT_CHANNELS = 1 // Número de canais da máscara de saída (1 para tons de cinza)

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            selectedBitmap = bitmap
            binding.imageView.setImageBitmap(bitmap)
            binding.imageViewSegmentation.setImageBitmap(null) // Limpa a máscara anterior
        }
    }

    private val takePictureLauncher = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
        if (bitmap != null) {
            selectedBitmap = bitmap
            binding.imageView.setImageBitmap(bitmap)
            binding.imageViewSegmentation.setImageBitmap(null) // Limpa a máscara anterior
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            openGallery()
        } else {
            Toast.makeText(this, "Permission denied to read your storage.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (segmenter == null) {
            segmenter = Segmenter(assets, "pet_segmentation_model_int8.tflite", IMAGE_SIZE, OUTPUT_CHANNELS)
        }

        binding.btnCamera.setOnClickListener {
            takePictureLauncher.launch(null)
        }

        binding.btnGallery.setOnClickListener {
            if (android.os.Build.VERSION.SDK_INT <= android.os.Build.VERSION_CODES.S_V2) {
                requestPermissionLauncher.launch(android.Manifest.permission.READ_EXTERNAL_STORAGE)
            } else {
                openGallery()
            }
        }

        binding.btnPredict.setOnClickListener {
            runSegmentation()
        }
    }

    private fun openGallery() {
        pickImageLauncher.launch("image/*")
    }

    private fun runSegmentation() {
        val bitmap = selectedBitmap ?: run {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show()
            return
        }
        val segmentationMask = segmenter?.segment(bitmap)

        if (segmentationMask != null) {
            binding.imageViewSegmentation.setImageBitmap(segmentationMask)
            Toast.makeText(this, "Segmentation successful!", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Segmentation failed!", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onStop() {
        super.onStop()
        segmenter?.close()
    }
}
