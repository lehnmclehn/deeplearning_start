package helper

import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import java.io.File

/**
 * import a dataset from an CSV file
 */
class CSVImporter {

    companion object {
        /**
         * extracts the label column from the raw data
         */
        private fun extractLabelColumn(
            data: Array<FloatArray>,
            labelColumnIdx: Int,
        ): FloatArray =
            data.map { it[labelColumnIdx] }.toFloatArray()

        /**
         * extracts the data without the label column from the raw data
         */
        private fun extractDataColumns(
            data: Array<FloatArray>,
            labelColumnIdx: Int,
        ): Array<FloatArray> =
            data.map {
                val list = it.toMutableList()
                list.removeAt(labelColumnIdx)
                list.toFloatArray()
            }.toTypedArray()

        private fun Array<FloatArray>.normalize(): Array<FloatArray> {
            println("Array<FloatArray>.normalize() -> start (${this.size},${this[0].size})")

            // we change the array from columns to lines and normalize values
            val reshapedArray = this[0].mapIndexed { idx, column ->
                val columnValues = this.map { line -> line[idx] }
                val minColumnValue = columnValues.minOrNull() ?: 0.0f
                val maxColumnValue = columnValues.maxOrNull() ?: 0.0f

                columnValues.map { (it - minColumnValue) / (maxColumnValue - minColumnValue) }
            }

            println("Array<FloatArray>.normalize() -> reshapedArray (${reshapedArray.size},${reshapedArray[0].size}) ")
            reshapedArray.take(10).forEach {
                it.take(10).forEach{value -> print("$value,")}
                println()
            }

            // now we bring the order back from columns to lines
            val result = reshapedArray[0].mapIndexed { idx, column ->
                reshapedArray.map { line -> line[idx] }.toFloatArray()
            }.toTypedArray()
            println("Array<FloatArray>.normalize() -> stop (${result.size},${result[0].size})")
            return result
        }

        fun loadAsFloatArray(fileName: String): Array<FloatArray> {
            val f = File(fileName)

            val result = f.readLines().mapIndexedNotNull { idx, line ->
                // header will be skipped
                when (idx) {
                    0 -> null
                    else -> line.split(";").map {
                        try {
                            it.toFloat()
                        } catch (e: Exception) {
                            0.0f
                        }
                    }.toFloatArray()
                }

            }.toTypedArray()
            return result
        }

        fun loadAsDataset(
            fileName: String,
            labelColumnIdx: Int,
        ): Dataset {
            val rawData = loadAsFloatArray(fileName)

            require(labelColumnIdx < rawData[0].size) { "Invalid column index $labelColumnIdx dataset" }

            // TODO: das kann weg
            println("ACHTUNG - TEMPORÄR umformatiert auf mehrere Label")
            /*
            .map {
                val category = (it /100000.0f).toInt().coerceAtMost(9)
                category.toFloat()
            }.toFloatArray()
             */
            val labels = extractLabelColumn(rawData, labelColumnIdx)
            val data = extractDataColumns(rawData, labelColumnIdx).normalize()

            val dataset = OnHeapDataset.create(data, labels)

            return dataset
        }
    }
}