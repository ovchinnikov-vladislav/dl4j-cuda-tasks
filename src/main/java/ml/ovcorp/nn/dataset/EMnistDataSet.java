package ml.ovcorp.nn.dataset;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class EMnistDataSet {

    private static final Logger log = LoggerFactory.getLogger(EMnistDataSet.class);

    public static DataSetIterator getTrains(int batchSize, int seed) throws IOException {
        log.info("Загрузка тренировочных данных EMNIST для обучения...");
        return new EmnistDataSetIterator(EmnistDataSetIterator.Set.BALANCED,  batchSize, true, seed);
    }

    public static DataSetIterator getTests(int batchSize, int seed) throws IOException {
        log.info("Загрузка тестовых данных EMNIST для обучения...");
        return new EmnistDataSetIterator(EmnistDataSetIterator.Set.BALANCED, batchSize, false, seed);
    }

}
