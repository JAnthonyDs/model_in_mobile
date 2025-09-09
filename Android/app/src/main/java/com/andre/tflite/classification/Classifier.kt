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

class Classifier(
    assetManager: AssetManager,
    modelPath: String,
    private val imageSize: Int = 128,
    private val numClasses: Int = 3
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

    fun classify(bitmap: Bitmap): FloatArray {
        val scaled = bitmap.scale(imageSize, imageSize, filter = true)
        val input = makeInputBuffer(scaled)

        val classProbs: FloatArray = when (outType) {
            DataType.FLOAT32 -> {
                val out = Array(1) { FloatArray(numClasses) }
                interpreter.run(input, out)
                out[0]
            }

            DataType.UINT8 -> {
                val out = Array(1) { ByteArray(numClasses) }
                interpreter.run(input, out)
                // (Optional) dequantize if you need probabilities: (q - zero) * scale
                (out[0].map { ((it.toInt() and 0xFF) - outZero) * outScale }.toFloatArray())
            }

            DataType.INT8 -> {
                val out = Array(1) { ByteArray(numClasses) }
                interpreter.run(input, out)
                (out[0].map { (it.toInt() - outZero) * outScale }.toFloatArray())
            }
            else -> throw IllegalArgumentException("Unsupported output type: $outType")
        }
        return classProbs
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
