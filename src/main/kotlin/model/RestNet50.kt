package model

import helper.FitCallback
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.history.BatchTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.EpochTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.TrainingHistory
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.model.resnet50Light
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.fashionMnist
import java.io.File
import java.util.*


private const val EPOCHS = 2
private const val TRAINING_BATCH_SIZE = 64
private const val TEST_BATCH_SIZE = 64
private const val NUM_CLASSES = 10
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L

class RestNet50 {
    private val modelPath = "./model/model2_resnet50"

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

    fun trainModel() {
        val (trainOrig, test) = fashionMnist()

        val (train, trainReserve) = trainOrig.split(0.1)

        resnet50Light(imageSize = IMAGE_SIZE, numberOfClasses = NUM_CLASSES, numberOfInputChannels = NUM_CHANNELS).use {

            println("Compiling model - RestNet50")

            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            println(
                "Fitting data: \n" +
                        "samples = ${train.xSize()}\n" +
                        "Badge   = $TRAINING_BATCH_SIZE\n" +
                        "RoundPerEpoch = ${train.xSize() / TRAINING_BATCH_SIZE}"
            )

            val start = System.currentTimeMillis()
            it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, callback = FitCallback())
            println("Training time: ${(System.currentTimeMillis() - start) / 1000f}")

            println("Testing accuracy")

            val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

            println("Accuracy: $accuracy")

            it.save(File(modelPath), writingMode = WritingMode.OVERRIDE)
        }
    }

    fun checkModel() {
        val (train, test) = fashionMnist()
        TensorFlowInferenceModel.load(File(modelPath)).use { model ->
            model.reshape(28, 28, 1)

            val (miniTest, test2) = test.split(0.1)
            val accuracy = model.evaluate(dataset = miniTest, metric = Metrics.ACCURACY)
            println("Accuracy: $accuracy")

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