import helper.BaseModel
import helper.ClassificationModel
import helper.RegressionModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import java.util.*


class ModelHouses : RegressionModel (
    dataSetFileName = "./cache/datasets/houses/housing_without_ocean.csv",
    labelColumnIndex = 8,
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

class ModelGer30 : ClassificationModel (
    dataSetFileName = "./cache/datasets/ger30_daily/export.csv",
    labelColumnIndex = 24,
    modelPath = "./model/model4_ger30_1",
) {
    override fun model(): Sequential =
        Sequential.of(
            Input(24),
            Dense(200, Activations.Relu),
            Dense(100, Activations.Relu),
            Dense(50, Activations.Relu),
            Dense(3, Activations.Relu)
        )

}

fun main() {
    val model = ModelGer30()
    //model.trainModel(10000)
     model.checkModel()
}