package helper

import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.history.BatchTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.EpochTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.TrainingHistory


/** Simple custom Callback object. */
class FitCallback : Callback() {
    override fun onEpochBegin(epoch: Int, logs: TrainingHistory) {
        // println("Epoch $epoch begins.")
    }

    override fun onEpochEnd(epoch: Int, event: EpochTrainingEvent, logs: TrainingHistory) {
        if (epoch % 10 == 0)
            println("Epoch $epoch ends -> LOSS=[${event.lossValue} | ${event.valLossValue}] " +
                    "METRIC=[${event.metricValues} | ${event.valMetricValues}]")
    }

    override fun onTrainBatchBegin(batch: Int, batchSize: Int, logs: TrainingHistory) {
        // println("Training batch $batch begins.")
    }

    override fun onTrainBatchEnd(batch: Int, batchSize: Int, event: BatchTrainingEvent, logs: TrainingHistory) {
        // println("Training batch $batch ends with loss ${event.lossValue}.")
    }

    override fun onTrainBegin() {
        println("Train begins")
    }

    override fun onTrainEnd(logs: TrainingHistory) {
        println("Train ends with last loss ${logs.lastBatchEvent().lossValue}")
    }
}
