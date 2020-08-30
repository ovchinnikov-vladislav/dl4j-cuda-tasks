package ml.ovcorp.nn.model.mnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class MnistDataSet {

    private static final Logger log = LoggerFactory.getLogger(MnistDataSet.class);

    public static DataSetIterator getTrains(int batchSize, int seed) throws IOException {
        log.info("Загрузка тренировочных данных MNIST для обучения...");
        return new MnistDataSetIterator(batchSize, true, seed);
    }

    public static DataSetIterator getTests(int batchSize, int seed) throws IOException {
        log.info("Загрузка тестовых данных MNIST для обучения...");
        return new MnistDataSetIterator(batchSize, false, seed);
    }

}
