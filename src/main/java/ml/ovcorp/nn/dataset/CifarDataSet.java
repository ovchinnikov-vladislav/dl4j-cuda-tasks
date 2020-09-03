package ml.ovcorp.nn.dataset;

import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class CifarDataSet {

    private static final Logger log = LoggerFactory.getLogger(CifarDataSet.class);

    public static DataSetIterator getTrains(int batchSize) throws IOException {
        log.info("Загрузка тренировочных данных CIFAR10 для обучения...");
        return new Cifar10DataSetIterator(batchSize, DataSetType.TRAIN);
    }

    public static DataSetIterator getTests(int batchSize) throws IOException {
        log.info("Загрузка тестовых данных CIFAR10 для обучения...");
        return new Cifar10DataSetIterator(batchSize, DataSetType.TEST);
    }

}
