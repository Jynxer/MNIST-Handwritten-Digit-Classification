// File: ReLU.java
// ReLU layer
package minet.layer;

import org.jblas.*;

import java.util.List;

/**
 * A class for ReLU layers {@literal y = max(0, x)}.
 *
 * @author Jordan Rowley
 */
public class ReLU implements Layer {
	
	// for backward
    DoubleMatrix Y;
    
    public ReLU() {}

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
        // Y[i] = max(0, X[i])

        DoubleMatrix Y = DoubleMatrix.zeros(X.rows, X.columns);

        for (int i = 0; i < X.length; i++) {
            if (X.get(i) > 0) {
                Y.put(i, X.get(i));
            }
        }

        this.Y = Y.dup();
        return Y;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        // X[i] = {1 if Y > 0, 0 if Y <= 0}

        DoubleMatrix out = DoubleMatrix.zeros(Y.rows, Y.columns);

        for (int i = 0; i < Y.length; i++) {
            if (Y.get(i) > 0) {
                out.put(i, 1);
            }
        }

        return gY.mul(out);
    }

    @Override
    public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights) {
        return weights;
    } 

    @Override
    public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> gradients) {
        return gradients;
    }

    @Override
    public String toString() {
        return "ReLU";
    }
}
