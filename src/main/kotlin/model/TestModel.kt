package model

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset


class TestModel {
    private val rawData = arrayOf(
        arrayOf(23, 25, 1),
        arrayOf(29, 21, 1),
        arrayOf(28, 24, 1),
        arrayOf(30, 21, 1),
        arrayOf(21, 29, 1),
        arrayOf(25, 25, 1),
        arrayOf(25, 29, 1),
        arrayOf(23, 27, 1),
        arrayOf(28, 27, 1),
        arrayOf(1, 5, 2),
        arrayOf(3, 2, 2),
        arrayOf(1, 2, 2),
        arrayOf(4, 1, 2),
        arrayOf(4, 3, 2),
        arrayOf(6, 1, 2),
        arrayOf(3, 4, 2),
        arrayOf(3, 2, 2),
        arrayOf(4, 2, 2),
        arrayOf(3, 2, 2),
        arrayOf(24, 23, 1),
        arrayOf(20, 21, 1),
        arrayOf(22, 28, 1),
        arrayOf(22, 21, 1),
        arrayOf(29, 29, 1),
        arrayOf(2, 1, 2),
        arrayOf(5, 2, 2),
        arrayOf(2, 5, 2),
        arrayOf(6, 2, 2),
        arrayOf(28, 21, 1),
        arrayOf(3, 3, 2),
    )

    private val train: Dataset
    private val test: Dataset

    init {
        val (data, label) = prepareData(rawData)
        val dataset = OnHeapDataset.create(data, label)
        val (tr, te) = dataset.split(0.5)
        train = tr
        test = te
    }

    fun prepareData(data: Array<Array<Int>>): Pair<Array<FloatArray>, FloatArray> {
        val myMap = data.map { line ->
            val l = line.toList()
            val vector = l.subList(0, 2).map { it.toFloat() / 100.0f }.toFloatArray()
            val label = l.last().toFloat()
            vector to label
        }.toMap()

        val result = Pair(
            myMap.keys.toTypedArray(),
            myMap.values.toFloatArray(),
        )
        return result
    }


    fun build() {
        val model = Sequential.of(
            Input(2),
            Flatten(),
            Dense(10),
            Dense(3)
        )

        model.use {

            println("Compiling model- Test")

            it.compile(
                optimizer = Adam(),
                loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
                metric = Metrics.ACCURACY
            )

            println("Fitting data")
            // You can think of the training process as "fitting" the model to describe the given data :)
            it.fit(
                dataset = train,
                epochs = 200,
                batchSize = 1,
            )

            println("Testing accuracy")

            val accuracy = it.evaluate(dataset = test, batchSize = 1).metrics[Metrics.ACCURACY]

            println("Accuracy: $accuracy")


            println("${model.predict(floatArrayOf(3.0f,2.0f))}")
            println("${model.predict(floatArrayOf(4.0f,1.0f))}")
            println("${model.predict(floatArrayOf(1.0f,5.0f))}")
            println("${model.predict(floatArrayOf(25.0f,21.0f))}")
            println("${model.predict(floatArrayOf(29.0f,25.0f))}")
        }
    }
}
