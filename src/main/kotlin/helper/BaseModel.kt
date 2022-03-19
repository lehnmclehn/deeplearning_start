package helper

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.printSummary
import org.jetbrains.kotlinx.dl.dataset.Dataset
import java.io.File
import java.io.FileNotFoundException

abstract class BaseModel(
    private val dataSetFileName: String,
    private val labelColumnIndex: Int,
    private val modelPath: String,
    private val losses: Losses,
    private val metrics: Metrics,
) {
    private val train: Dataset
    private val validate: Dataset
    private val test: Dataset

    init {
        val dataSet = CSVImporter.loadAsDataset(dataSetFileName, labelColumnIndex)
        println("Data read with: ${dataSet.xSize()} lines")
        println("Data read with: ${dataSet.toString()} lines")

        val (train, testTemp) = dataSet.split(0.6)
        val (validate, test) = testTemp.split(0.5)
        this.train = train
        this.validate = validate
        this.test = test

        println("Data train: ${train.xSize()} lines")
    }

    protected abstract fun model(): Sequential

    fun loadAsInferenceModel() {
        // TensorFlowInferenceModel.load(File(modelPath)).use { model ->
    }

    fun trainModel(
        epochs: Int = 100,
    ) {
        model().use {

            println("Compiling model- Test")

            it.compile(
                optimizer = Adam(),
                loss = losses,
                metric = metrics,
            )

            try {
                it.loadWeights(File(modelPath))
            } catch (e:FileNotFoundException) {}

            it.printSummary()

            println("Fitting data")
            it.fit(
                trainingDataset = train,
                validationDataset = validate,
                epochs = epochs,
                trainBatchSize = 100,
                validationBatchSize = 100,
                callback = FitCallback(),
            )

            println(metrics.name)
            val metricValue = it.evaluate(dataset = test, batchSize = 100).metrics[metrics]
            println("${metrics.name}: $metricValue")

            it.save(File(modelPath), writingMode = WritingMode.OVERRIDE)
        }
    }

    fun checkModel() {
        model().use { model ->
            model.compile(
                optimizer = Adam(),
                loss = losses,
                metric = metrics,
            )

            model.loadWeights(File(modelPath))
            test.shuffle()
            (1..50).forEach { idx ->
                val predict = model.predictSoftly(test.getX(idx))
                val diff = test.getY(idx) - predict[0]
                println("Predicted: ${predict.contentToString()} --> ${test.getY(idx)} --> $diff")
            }
        }
    }
}
