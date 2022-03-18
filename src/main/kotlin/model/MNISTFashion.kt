package model

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.fashionMnist
import java.io.File
import java.util.*

class MNISTFashion {
    private val modelPath = "./model/model1_simple"

    val stringLabels = mapOf(
        0 to "T-shirt/top",
        1 to "Trousers",
        2 to "Pullover",
        3 to "Dress",
        4 to "Coat",
        5 to "Sandals",
        6 to "Shirt",
        7 to "Sneakers",
        8 to "Bag",
        9 to "Ankle boots"
    )

    val model = Sequential.of(
        Input(28, 28, 1),
        Flatten(),
        Dense(300),
        Dense(100),
        Dense(10)
    )

    fun trainModel() {
        val (train, test) = fashionMnist()

        model.use {

            println("Compiling model- model.MNISTFashion")

            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            println("Fitting data")
            // You can think of the training process as "fitting" the model to describe the given data :)
            it.fit(
                dataset = train,
                epochs = 10,
                batchSize = 100
            )

            println("Testing accuracy")

            val accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

            println("Accuracy: $accuracy")
            it.save(File(modelPath), writingMode = WritingMode.OVERRIDE)
        }
    }

    fun checkModel() {
        val (train, test) = fashionMnist()
        TensorFlowInferenceModel.load(File(modelPath)).use { model ->
            model.reshape(28, 28, 1)
            val r = Random()
            (1..10).forEach {
                val idx = r.nextInt(10000)
                val prediction = model.predict(test.getX(idx))
                val actualLabel = test.getY(idx)

                println("$prediction => $actualLabel. This corresponds to class ${stringLabels[prediction]}")
            }
        }
    }

}