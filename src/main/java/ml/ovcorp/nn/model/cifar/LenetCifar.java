package ml.ovcorp.nn.model.cifar;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LenetCifar {

    private static final Logger log = LoggerFactory.getLogger(LenetCifar.class);

    public static ComputationGraphConfiguration getLenetConf(int classes, int height, int width, int channels, int seed) {
        log.info("Построение модели нейронной сети Lenet...");
        log.info("Количество классов: {}", classes);
        log.info("Высота объектов: {}", height);
        log.info("Ширина объектов: {}", width);
        log.info("Ширина канала (цвет изображения): {}", channels);

        return new NeuralNetConfiguration.Builder()
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .seed(seed)
                .l2(0.005)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutionalFlat(height, width, channels))
                .addLayer("cnn1",
                        new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                                .name("cnn1")
                                .nIn(channels)
                                .nOut(50)
                                .biasInit(0)
                                .build(), "input")
                .addLayer("maxpool1",
                        new SubsamplingLayer.Builder(new int[]{2, 2}, new int[]{2, 2})
                                .name("maxpool1")
                                .build(), "cnn1")
                .addLayer("cnn2",
                        new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{5, 5}, new int[]{1, 1})
                                .name("cnn2")
                                .nOut(100)
                                .biasInit(0)
                                .build(), "maxpool1")
                .addLayer("maxpool2",
                        new SubsamplingLayer.Builder(new int[]{2, 2}, new int[]{2, 2})
                                .name("maxool2")
                                .build(), "cnn2")
                .addLayer("denseLayer",
                        new DenseLayer.Builder()
                                .nOut(500)
                                .build(), "maxpool2")
                .addLayer("outputLayer",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(classes)
                                .activation(Activation.SOFTMAX)
                                .build(), "denseLayer")
                .setOutputs("outputLayer")
                .build();
    }

}
