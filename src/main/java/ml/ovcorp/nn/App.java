package ml.ovcorp.nn;

import ml.ovcorp.nn.enums.DataSetType;
import ml.ovcorp.nn.enums.NeuralNetworkType;
import ml.ovcorp.nn.model.cifar.CapsNetCifar;
import ml.ovcorp.nn.model.cifar.CifarDataSet;
import ml.ovcorp.nn.model.cifar.LenetCifar;
import ml.ovcorp.nn.model.emnist.CapsNetEMnist;
import ml.ovcorp.nn.model.emnist.EMnistDataSet;
import ml.ovcorp.nn.model.emnist.LenetEMnist;
import ml.ovcorp.nn.model.mnist.CapsNetMnist;
import ml.ovcorp.nn.model.mnist.LenetMnist;
import ml.ovcorp.nn.model.mnist.MnistDataSet;
import ml.ovcorp.nn.util.Utils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.fetchers.EmnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.time.LocalDateTime;

public class App {

    private static final Logger log = LoggerFactory.getLogger(App.class);

    public static void main(String... args) throws Exception {
        NeuralNetworkType neuralNetworkType = NeuralNetworkType.LENET;
        DataSetType dataSetType = DataSetType.MNIST;
        if (args.length == 2) {
            neuralNetworkType = NeuralNetworkType.valueOf(args[0]);
            dataSetType = DataSetType.valueOf(args[1]);
        } else {
            log.info("Default settings: Network - {}, Dataset - {}", neuralNetworkType, dataSetType);
        }

        int batchSize = 28; // Размер пакета выборки
        int nEpochs = 1; // Количество тренеровочных эпох
        int seed = 12345; // Коэффициент рассеивания данных

        // Create an iterator using the batch size for one iteration
        DataSetIterator train = null;
        DataSetIterator test = null;
        ComputationGraph net = null;
        switch (dataSetType) {
            case MNIST:
                train = MnistDataSet.getTrains(batchSize, seed);
                test = MnistDataSet.getTests(batchSize, seed);
                log.info("Построение модели...");
                net = getNetworkMnist(neuralNetworkType, seed);
                break;
            case EMNIST:
                train = EMnistDataSet.getTrains(batchSize, seed);
                test = EMnistDataSet.getTests(batchSize, seed);
                log.info("Построение модели...");
                net = getNetworkEMnist(neuralNetworkType, seed);
                break;
            case CIFAR:
                train = CifarDataSet.getTrains(batchSize);
                test = CifarDataSet.getTests(batchSize);
                log.info("Построение модели...");
                net = getNetworkCifar(neuralNetworkType, seed);
                break;
            default:
                log.error("Набор данных неопределен. Будет произведен выход из программы.");
                System.exit(0);
        }

        net.init();

        log.info("Конфигурация модели:\n{}", net.getConfiguration().toJson());
        log.info("Количество параметров модели: {}", net.numParams());
        log.info("Анализ информации о слоях.");
        int i = 0;
        for (Layer l : net.getLayers()) {
            log.info("{}. Тип слоя: {}. Количество параметров слоя: {}.",
                    ++i,
                    l.type(),
                    l.numParams());
        }

        log.info("Старт обучения...");
        LocalDateTime start = LocalDateTime.now();
        log.info("Время начала обучения: {}", start);
        net.setListeners(new ScoreIterationListener(1), new EvaluativeListener(test, 1, InvocationType.EPOCH_END)); //Print score every 10 iterations and evaluate on test set every epoch
        net.fit(train, nEpochs);
        LocalDateTime end = LocalDateTime.now();
        log.info("Время начала обучения: {}", start);
        log.info("Время конца обучения: {}", end);
        log.info("Количество эпох: {}", nEpochs);
        Utils.ResultTime resultTime = Utils.diffLocalDateTime(start, end);
        log.info("Общее время обучения составило: {}", resultTime);

        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "template_model.zip");

        log.info("Сохранения модели: {}", path);
        net.save(new File(path), true);

        log.info("Завершение работы программы.");
    }

    private static ComputationGraph getNetworkMnist(NeuralNetworkType neuralNetworkType, int seed) {
        switch (neuralNetworkType) {
            case LENET:
                return new ComputationGraph(LenetMnist.getLenetConf(10, 28, 28, 1, seed));
            case CAPSNET:
                return new ComputationGraph(CapsNetMnist.getCapsNetConf(10, 28, 28, 1, seed));
            default:
                log.error("Нейронная сеть не определена. Будет произведен выход из программы.");
                System.exit(0);
        }
        return null;
    }

    private static ComputationGraph getNetworkEMnist(NeuralNetworkType neuralNetworkType, int seed) {
        switch (neuralNetworkType) {
            case LENET:
                return new ComputationGraph(LenetEMnist.getLenetConf(28, 28, 1, seed));
            case CAPSNET:
                return new ComputationGraph(CapsNetEMnist.getCapsNetConf(28, 28, 1, seed));
            default:
                log.error("Нейронная сеть не определена. Будет произведен выход из программы.");
                System.exit(0);
        }
        return null;
    }

    private static ComputationGraph getNetworkCifar(NeuralNetworkType neuralNetworkType, int seed) {
        switch (neuralNetworkType) {
            case LENET:
                return new ComputationGraph(LenetCifar.getLenetConf(10, 32, 32, 3, seed));
            case CAPSNET:
                return new ComputationGraph(CapsNetCifar.getCapsNetConf(10, 32, 32, 3, seed));
            default:
                log.error("Нейронная сеть не определена. Будет произведен выход из программы.");
                System.exit(0);
        }
        return null;
    }
}
