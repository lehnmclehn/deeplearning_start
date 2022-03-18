import helper.CSVImporter
import helper.FitCallback
import model.MNISTFashion
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.api.core.summary.printSummary
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.fashionMnist
import java.io.File
import java.util.*

abstract class BaseModel(
    private val dataSetFileName: String,
    private val modelPath: String,
) {
    private val train: Dataset
    private val validate: Dataset
    private val test: Dataset

    init {
        val dataSet = CSVImporter.loadAsDataset(dataSetFileName, 8)
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
        epochs : Int = 100,
    ) {
        model().use {

            println("Compiling model- Test")

            it.compile(
                optimizer = Adam(),
                loss = Losses.MAE,
                metric = Metrics.MAE,
            )

            it.loadWeights(File(modelPath))

            it.printSummary()

            println("Fitting data")
            // You can think of the training process as "fitting" the model to describe the given data :)
            it.fit(
                trainingDataset = train,
                validationDataset = validate,
                epochs = epochs,
                trainBatchSize = 100,
                validationBatchSize = 100,
                callback = FitCallback(),
            )

            println("MAE")
            val accuracy = it.evaluate(dataset = test, batchSize = 10000).metrics[Metrics.MAE]
            println("MAE: $accuracy")

            it.save(File(modelPath), writingMode = WritingMode.OVERRIDE)
        }
    }

    fun checkModel() {
        model().use { model ->
            model.loadWeights(File(modelPath))
            test.shuffle()
            (1..10).forEach { idx ->
                val predict = model.predictSoftly(test.getX(idx))
                val diff = test.getY(idx) - predict[0]
                println("Predicted: ${predict.contentToString()} --> ${test.getY(idx)} --> $diff")
            }
        }
    }
}

class ModelHouses : BaseModel(
    dataSetFileName = "./cache/datasets/houses/housing_without_ocean.csv",
    modelPath = "./model/model3_houses",
) {
    override fun model(): Sequential =
        Sequential.of(
            Input(8),
            Dense(200, Activations.Relu),
            Dense(100, Activations.Relu),
            Dense(50, Activations.Relu),
            Dense(1, Activations.Linear)
        )

}

fun main() {
    val model = ModelHouses()
    model.trainModel()
    model.checkModel()
}