import minet.Dataset;
import minet.layer.*;
import minet.layer.Linear.WeightInitXavier;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import minet.optim.Optimizer;
import minet.optim.SGD;
import minet.util.Pair;
import org.jblas.DoubleMatrix;
import org.jblas.util.Logger;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Random;


public class Part1 {

    public static void train(Layer net, Loss loss, Optimizer optimizer, Dataset traindata, Dataset devdata, int batchsize, int nEpochs, int patience, Random rnd) {

        System.out.println("Training...");

        int notAtPeak = 0; // number of epochs since last peak
        double peakAcc = -1; // the best accuracy so far
        double totalLoss = 0; // the total loss of the current epoch
        double lossVal; // the loss value at each epoch

        for (int e = 0; e < nEpochs; e++) {

            System.out.printf("\nepoch %d:\n", e);
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

            System.out.printf("total loss: %.6f\n", totalLoss);

            // check if accuracy of devdata is the same of lower for 'patience' number of times
            double acc = eval(net, devdata, batchsize);

            System.out.printf("accuracy: %.6f\n", acc);

            if (acc > peakAcc) {
                peakAcc = acc;
                notAtPeak = 0;
            } else {
                notAtPeak = notAtPeak + 1;
                System.out.printf("Not at peak " + notAtPeak + " times consecutively");
            }

            if (notAtPeak >= patience) {
                //System.out.println("== Stopped Training at Epoch " + e + " ==\n");
                break;
            }
            
        }

        System.out.println("\ntraining is finished");

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
        System.out.println("Loading data...");
        Dataset trainset = Dataset.loadTxt(args[1]);
        Dataset devset = Dataset.loadTxt(args[2]);
        Dataset testset = Dataset.loadTxt(args[3]);
        //System.out.printf("train: %d instances\n", trainset.getSize());
        //System.out.printf("dev: %d instances\n", devset.getSize());
        //System.out.printf("test: %d instances\n", testset.getSize());

        // creates network
        System.out.println("\nCreating network...");
        int indims = trainset.getInputDims(); // gets the dimemsions of x
        int hiddims = 1000;
        int outdims = 10;
        Sequential net = new Sequential(new Layer[] {
            new Linear(indims, hiddims, new WeightInitXavier()),
            new Sigmoid(), // activation function
            new Linear(hiddims, outdims, new WeightInitXavier()),
            new Softmax()
        });

        // cross entropy loss function to be passed into train function
        CrossEntropy loss = new CrossEntropy();

        // default = 1.0
        double learningRate = 1.0;
        // stochastic gradient decent object
        Optimizer sgd = new SGD(net, learningRate);

        int batchsize = 1000;
        int nEpochs = 50;
        int patience = 5;

        // calls the train function
        train(net, loss, sgd, trainset, devset, batchsize, nEpochs, patience, rnd);

        double testAcc = eval(net, testset, batchsize);

        System.out.println("accuracy on test set: " + testAcc);

    }
}
