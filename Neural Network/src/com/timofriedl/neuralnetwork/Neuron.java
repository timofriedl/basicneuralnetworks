package com.timofriedl.neuralnetwork;

/**
 * Represents one neuron of a {@link Layer} in a {@link DNN}.
 * 
 * @author Timo Friedl
 */
public class Neuron {

	/**
	 * the option to let this neuron be a bias neuron
	 */
	private final boolean bias;

	/**
	 * the current value of this neuron
	 */
	private double value;

	/**
	 * the connections to this neuron
	 */
	private final Connection[] incomingConnections;

	/**
	 * Creates a new neuron of a {@link Layer} in a {@link DNN}.
	 * 
	 * @param bias      the option to let this neuron be a bias neuron
	 * @param prevLayer the layer before this neurons layer or null if this neuron
	 *                  is in the first layer
	 */
	public Neuron(boolean bias, Layer prevLayer) {
		this.bias = bias;

		if (bias) {
			value = 1.0;
			incomingConnections = new Connection[0];
		} else {
			incomingConnections = new Connection[prevLayer == null ? 0 : prevLayer.size()];

			for (int i = 0; i < incomingConnections.length; i++)
				incomingConnections[i] = new Connection(prevLayer.get(i), this, prevLayer.size());
		}
	}

	/**
	 * Updates the value of this {@link Neuron} by adding the weighted values of the
	 * neurons of the previous {@link Layer}.
	 */
	public void update() {
		if (bias)
			return;

		double newValue = 0.0;
		for (int i = 0; i < incomingConnections.length; i++)
			newValue += incomingConnections[i].getWeight() * incomingConnections[i].getFrom().value;

		value = relu(newValue);
	}

	/**
	 * The rectified linear activation function.
	 * 
	 * @param input the input value
	 * @return zero if <code>input</code> is negative, the identity value else
	 */
	private double relu(double input) {
		return Math.max(0, input);
	}

	/**
	 * @return true if and only if this {@link Neuron} is a bias neuron.
	 */
	public boolean isBias() {
		return bias;
	}

	/**
	 * @return the current value of this {@link Neuron}
	 */
	public double getValue() {
		return value;
	}

	/**
	 * @param value the new value of this {@link Neuron}
	 */
	public void setValue(double value) {
		this.value = value;
	}

	/**
	 * @return the connections to this {@link Neuron}
	 */
	public Connection[] getIncomingConnections() {
		return incomingConnections;
	}

}
