import minet.Dataset;
import minet.layer.*;
import minet.layer.Linear.WeightInitXavier;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import minet.loss.MeanSquaredError;
import minet.optim.Optimizer;
import minet.optim.SGD;
import minet.util.Pair;
import org.jblas.DoubleMatrix;
import org.jblas.util.Logger;

import java.io.IOException;
import java.util.Random;


public class Part3 {

    public static void train(Layer net, Loss loss, Optimizer optimizer, Dataset traindata, Dataset devdata, int batchsize, int nEpochs, int patience, Random rnd) {

        // System.out.println("Training...");

        int notAtPeak = 0; // number of epochs since last peak
        double peakAcc = -1; // the best accuracy so far
        double totalLoss = 0; // the total loss of the current epoch
        double lossVal; // the loss value at each epoch

        for (int e = 0; e < nEpochs; e++) {

            // System.out.printf("\nepoch %d:\n", e);
            // always shuffle the data between each epoch
            traindata.shuffle(rnd);
            // loss function value on evaluated examples in this epoch
            totalLoss = 0;

            while (true) {

                // get the next mini-batch
                Pair<DoubleMatrix> batch = traindata.getNextMiniBatch(batchsize);
                if (batch == null) {
                    break;
                }

                // always reset the gradients before performing backward
                optimizer.resetGradients();

                DoubleMatrix Yhat = net.forward(batch.first);

                // calculate the loss value then save it to a variable called lossVal
                lossVal = loss.forward(batch.second, Yhat);
                //System.out.println("Loss Value: " + lossVal);

                // calculate the network weights' gradients using backprop
                net.backward(loss.backward());

                // update network weights using the calculated gradients
                optimizer.updateWeights();

                // update totalLoss
                totalLoss = totalLoss + lossVal;

            }

            // System.out.printf("total loss: %.6f\n", totalLoss);

            // check if accuracy of devdata is the same of lower for 'patience' number of times
            double acc = eval(net, devdata, batchsize);

            // System.out.printf("accuracy: %.6f\n", acc);

            if (acc > peakAcc) {
                peakAcc = acc;
                notAtPeak = 0;
            } else {
                notAtPeak = notAtPeak + 1;
                // System.out.printf("Not at peak " + notAtPeak + " times consecutively");
            }

            if (notAtPeak >= patience) {
                //System.out.println("== Stopped Training at Epoch " + e + " ==\n");
                break;
            }
            
        }

        // System.out.println("\ntraining is finished");

    }

    public static double eval(Layer net, Dataset data, int batchsize) {

        data.reset(); // move pointer to beginning of dataset
        double correct = 0; // for counting how many predictions are correct
        int size = 0;

        // processing each mini-batch
        while(true) {
            // gets the next mini-batch
            Pair<DoubleMatrix> batch = data.getNextMiniBatch(batchsize);

            //stop iterating when no mini-batches left
            if (batch == null) {
                break;
            }

            // perform forward to compute the prediction values
            // each row of Yhat corresponds to the prediction for an input vector
            DoubleMatrix Yhat = net.forward(batch.first);

            // stores the number of rows in Yhat (also the number of elements)
            size = batch.second.rows;

            // counts how many predictions are correct
            for (int i = 0; i < size; i++) {
                if (batch.second.get(i) == Yhat.rowArgmaxs()[i]) {
                    correct = correct + 1.0;
                }
            }

        }

        // computes accuracy
        double acc = correct / data.getSize();

        return(acc);

    }

    public static void main(String[] args) throws IOException {
        // set initial random seed
        org.jblas.util.Random.seed(Integer.parseInt(args[0]));
        Random rnd = new Random(Integer.parseInt(args[0]));

        // disables jblas debug messages
        Logger.getLogger().setLevel(Logger.WARNING);

        // read datasets
        // System.out.println("Loading data...");
        Dataset trainset = Dataset.loadTxt(args[1]);
        Dataset devset = Dataset.loadTxt(args[2]);
        Dataset testset = Dataset.loadTxt(args[3]);
        //System.out.printf("train: %d instances\n", trainset.getSize());
        //System.out.printf("dev: %d instances\n", devset.getSize());
        //System.out.printf("test: %d instances\n", testset.getSize());

        // creates network
        // System.out.println("\nCreating network...");
        int indims = trainset.getInputDims(); // gets the dimemsions of x
        int hiddims = 1000;
        int outdims = 10;

        int batchsize = 1000;
        int nEpochs = 50;
        int patience = 5;
        double testAcc;

        Layer[] activationFunctions = {new ReLU(), new TanH()};
        int[] batchSizes = {500, 1000, 1500};
        double[] learningRates = {0.25, 0.5, 0.75, 1.0};
        Loss[] lossFunctions = {new CrossEntropy(), new MeanSquaredError()};

        for (int i = 0; i < activationFunctions.length; i++) {

            Sequential net = new Sequential(new Layer[] {
                new Linear(indims, hiddims, new WeightInitXavier()),
                activationFunctions[i], // activation function
                new Linear(hiddims, outdims, new WeightInitXavier()),
                new Softmax()
            });

            for (int j = 0; j < batchSizes.length; j++) {

                for (int k = 0; k < learningRates.length; k++) {

                    // stochastic gradient decent object
                    Optimizer sgd = new SGD(net, learningRates[k]);

                    for (int l = 0; l < lossFunctions.length; l++) {

                        // trains the network
                        train(net, lossFunctions[l], sgd, trainset, devset, batchSizes[j], nEpochs, patience, rnd);
                        // calculates the accuracy of the trained network on the test set
                        testAcc = eval(net, testset, batchsize);
                        //System.out.println("Using " + activationFunctions[i].toString() + "as the activation function, " + Integer.toString(batchSizes[j]) + " as the batch size, " + Double.toString(learningRates[k]) + "as the learning rate, and " + lossFunctions[l].toString() + "as the loss function, the test accuracy was " + testAcc);
                        System.out.println(activationFunctions[i].toString() + " " + Integer.toString(batchSizes[j]) + " " + Double.toString(learningRates[k]) + " " + lossFunctions[l].toString() + " " + Double.toString(testAcc));

                    }

                }

            }

        }

       

    }
}
