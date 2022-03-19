package helper

import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics

abstract class ClassificationModel(
    dataSetFileName: String,
    labelColumnIndex: Int,
    modelPath: String,
) : BaseModel(
    dataSetFileName = dataSetFileName,
    labelColumnIndex = labelColumnIndex,
    modelPath = modelPath,
    losses = Losses.BINARY_CROSSENTROPY,
    metrics = Metrics.ACCURACY,
) {
}