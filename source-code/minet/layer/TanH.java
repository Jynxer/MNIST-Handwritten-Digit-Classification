// File: TanH.java
// TanH layer
package minet.layer;

import org.jblas.*;

import java.util.List;

/**
 * A class for TanH layers {@literal y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))}.
 *
 * @author Jordan Rowley
 */
public class TanH implements Layer {
	
	// for backward
    DoubleMatrix Y;
    
    public TanH() {}

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
        // Y[i] = (exp(X[i]) - exp(-X[i])) / (exp(X[i]) + exp(-X[i]))
        DoubleMatrix Y = MatrixFunctions.tanh(X);

        this.Y = Y.dup();
        return Y;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        // X[i] = 1 - tanh²(Y[i])
        return (gY.mul((this.Y.mul(this.Y)).rsub(1))); 
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
        return "TanH";
    }
}
