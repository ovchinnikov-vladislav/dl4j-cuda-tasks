package ml.ovcorp.nn.model.mnist;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CapsNetMnist {

    private static final Logger log = LoggerFactory.getLogger(CapsNetMnist.class);

    public static ComputationGraphConfiguration getCapsNetConf(int classes, int height, int width, int channels, int seed) {
        log.info("Построение модели нейронной сети CapsNet...");
        log.info("Количество классов: {}", classes);
        log.info("Высота объектов: {}", height);
        log.info("Ширина объектов: {}", width);
        log.info("Ширина канала (цвет изображения): {}", channels);

        return new NeuralNetConfiguration.Builder()
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .seed(seed)
                .updater(new Adam())
                .graphBuilder()
                .addInputs("input")
                // Input Image
                .setInputTypes(InputType.convolutionalFlat(height, width, channels))
                // 1. Convolutional Layer - Start Encoder
                .addLayer("cnn", new ConvolutionLayer.Builder()
                        .nOut(256)
                        .kernelSize(9, 9)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build(), "input")
                // 2. Primary Capsules
                .addLayer("primary_capsules", new PrimaryCapsules.Builder(8, 32)
                        .kernelSize(9, 9)
                        .stride(2, 2)
                        .build(), "cnn")
                // 3. Digital Capsules - End Encoder
                .addLayer("digit_capsules", new CapsuleLayer.Builder(classes, 16, 1).build(), "primary_capsules")
                // 4. Start Decoder
                .addLayer("decoder1", new CapsuleStrengthLayer.Builder().build(), "digit_capsules")
                .addLayer("decoder2", new ActivationLayer.Builder(new ActivationSoftmax()).build(), "decoder1")
                .addLayer("decoder3", new LossLayer.Builder(new LossNegativeLogLikelihood()).build(), "decoder2")
                .setOutputs("decoder3")
                .build();
    }

}
