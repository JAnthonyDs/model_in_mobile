package com.andre.tflite.classification

import android.content.res.AssetManager
import android.graphics.Bitmap
import androidx.core.graphics.scale
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class Segmenter(
    assetManager: AssetManager,
    modelPath: String,
    private val imageSize: Int = 128, // Input image size (e.g., 256 for 256x256)
    private val outputChannels: Int = 1 // Number of channels in the output mask (e.g., 1 for grayscale, N for N classes)
) {

    private var interpreter: Interpreter
    private var nnapiDelegate: NnApiDelegate? = null
    private var gpuDelegate: GpuDelegate? = null

    // Input tensor info
    private var inType: DataType
    private var inScale = 1f
    private var inZero = 0

    // Output tensor info
    private var outType: DataType
    private var outScale = 1f
    private var outZero = 0
    private var outputShape: IntArray // Store the output shape

    init {
        val modelBuffer = loadModelFile(assetManager, modelPath)
        interpreter = buildInterpreterWithFallbacks(modelBuffer)

        // Cache I/O tensor metadata
        val inTensor = interpreter.getInputTensor(0)
        inType = inTensor.dataType()
        inTensor.quantizationParams().let {
            inScale = it.scale
            inZero = it.zeroPoint
        }

        val outTensor = interpreter.getOutputTensor(0)
        outType = outTensor.dataType()
        outTensor.quantizationParams().let {
            outScale = it.scale
            outZero = it.zeroPoint
        }
        outputShape = outTensor.shape() // Get the full output shape
    }

    private fun buildInterpreterWithFallbacks(modelBuffer: ByteBuffer): Interpreter {
        // Try NNAPI → GPU → CPU
        fun tryWith(options: Interpreter.Options.() -> Unit): Interpreter? = try {
            val opts = Interpreter.Options().apply { numThreads =
                Runtime.getRuntime().availableProcessors().coerceAtMost(4) }
            opts.options()
            Interpreter(modelBuffer, opts)
        } catch (_: Throwable) { null }

        // 1) NNAPI
        tryWith {
            nnapiDelegate = NnApiDelegate()
            addDelegate(nnapiDelegate)
        }?.let {
            println("Using NNAPI delegate")
            return it}

        // 2) GPU
        tryWith {
            gpuDelegate = GpuDelegate()
            addDelegate(gpuDelegate)
        }?.let {
            println("Using GPU delegate")
            return it }

        // 3) CPU
        println("Using CPU")
        return Interpreter(modelBuffer, Interpreter.Options().apply {
            numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
        })
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): ByteBuffer {
        val fd = assetManager.openFd(modelPath)
        FileInputStream(fd.fileDescriptor).channel.use { ch ->
            return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    fun segment(bitmap: Bitmap): Bitmap {
        val scaled = bitmap.scale(imageSize, imageSize, filter = true)
        val input = makeInputBuffer(scaled)

        // Assuming output is a single-channel mask or multi-channel probability map
        // The outputShape will tell us the actual dimensions.
        // For simplicity, let's assume the output is a [1, height, width, channels] tensor
        val outputHeight = outputShape[1]
        val outputWidth = outputShape[2]
        val outputNumChannels = outputShape[3]

        val outputByteBuffer = ByteBuffer.allocateDirect(outputHeight * outputWidth * outputNumChannels * when(outType) {
            DataType.FLOAT32 -> 4 // 4 bytes per float
            DataType.UINT8, DataType.INT8 -> 1 // 1 byte per byte
            else -> throw IllegalArgumentException("Unsupported output type: $outType")
        }).order(ByteOrder.nativeOrder())


        interpreter.run(input, outputByteBuffer)

        // Post-processing: Convert ByteBuffer to Bitmap
        return postProcessOutput(outputByteBuffer, outputWidth, outputHeight, outputNumChannels)
    }

    private fun postProcessOutput(outputBuffer: ByteBuffer, width: Int, height: Int, numChannels: Int): Bitmap {
        outputBuffer.rewind() // Rewind buffer to read from the beginning
        val pixels = IntArray(width * height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                // Assuming a single channel output for a grayscale mask
                // If it's multi-channel, you'd need to decide how to combine them (e.g., argmax for class ID)
                val pixelValue = when (outType) {
                    DataType.FLOAT32 -> {
                        val value = outputBuffer.float // Read float
                        // Dequantize if necessary for visualization
                        ((value * 255).roundToInt()).coerceIn(0, 255)
                    }
                    DataType.UINT8 -> {
                        val value = outputBuffer.get().toInt() and 0xFF // Read unsigned byte
                        // Dequantize if necessary for visualization: (q - zero) * scale
                        ((value - outZero) * outScale * 255).roundToInt().coerceIn(0, 255)
                    }
                    DataType.INT8 -> {
                        val value = outputBuffer.get().toInt() // Read signed byte
                        // Dequantize if necessary for visualization: (q - zero) * scale
                        ((value - outZero) * outScale * 255).roundToInt().coerceIn(0, 255)
                    }
                    else -> 0 // Default to black
                }

                // Create a grayscale pixel (ARGB)
                pixels[y * width + x] = 0xFF000000.toInt() or (pixelValue shl 16) or (pixelValue shl 8) or pixelValue
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }


    private fun makeInputBuffer(bitmap: Bitmap): ByteBuffer {
        return when (inType) {
            DataType.FLOAT32 -> bitmapToFloat32(bitmap)
            DataType.UINT8 -> bitmapToQuant(bitmap, signed = false)
            DataType.INT8 -> bitmapToQuant(bitmap, signed = true)
            else -> throw IllegalArgumentException("Unsupported input type: $inType")
        }
    }

    private fun bitmapToFloat32(bmp: Bitmap): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(imageSize * imageSize)
        bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        var i = 0
        repeat(imageSize * imageSize) {
            val p = pixels[i++]
            buf.putFloat(((p shr 16) and 0xFF) / 255f)
            buf.putFloat(((p shr 8) and 0xFF) / 255f)
            buf.putFloat((p and 0xFF) / 255f)
        }
        return buf
    }

    private fun bitmapToQuant(bmp: Bitmap, signed: Boolean): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(imageSize * imageSize * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(imageSize * imageSize)
        bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)

        // Quantization: q = clamp(round(x/scale) + zeroPoint)
        // with x in [0,1]
        val minQ = if (signed) -128 else 0
        val maxQ = if (signed) 127 else 255

        var i = 0
        repeat(imageSize * imageSize) {
            val p = pixels[i++]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f

            fun q(v: Float): Byte {
                val q = (v / inScale + inZero).roundToInt()
                return max(minQ, min(maxQ, q)).toByte()
            }

            buf.put(q(r)); buf.put(q(g)); buf.put(q(b))
        }
        return buf
    }

    fun close() {
        interpreter.close()
        nnapiDelegate?.close()
        gpuDelegate?.close()
    }
}