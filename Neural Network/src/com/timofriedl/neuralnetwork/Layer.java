package com.timofriedl.neuralnetwork;

import java.io.Serializable;

/**
 * Represents one layer in a {@link DNN}.
 * 
 * @author Timo Friedl
 */
public class Layer implements Serializable {

	/**
	 * SVUID
	 */
	private static final long serialVersionUID = -6606713589191062699L;

	/**
	 * the {@link Neuron}s of this layer
	 */
	private final Neuron[] neurons;

	/**
	 * the existence of a bias {@link Neuron} in this layer
	 */
	private final boolean bias;

	/**
	 * Creates a new layer of a {@link DNN}.
	 * 
	 * @param size      the number of non-bias neurons in this layer
	 * @param addBias   the option to create an additional bias {@link Neuron} in
	 *                  this layer
	 * @param prevLayer the previous layer or null if this is the first layer
	 */
	public Layer(int size, boolean addBias, Layer prevLayer) {
		this.bias = addBias;

		neurons = new Neuron[size + (bias ? 1 : 0)];

		for (int i = 0; i < neurons.length; i++)
			neurons[i] = new Neuron(bias && i == size, prevLayer);
	}

	/**
	 * Calculates all {@link Neuron} values in this layer with the values from the
	 * previous {@link Layer}.
	 */
	public void update() {
		for (int i = 0; i < neurons.length; i++)
			neurons[i].update();
	}

	/**
	 * Collects all non-bias values of this {@link Layer} to one array
	 * 
	 * @return the neuron values of this layer
	 */
	public double[] values() {
		final double[] values = new double[neurons.length - (bias ? 1 : 0)];

		for (int i = 0; i < values.length; i++)
			values[i] = get(i).getValue();

		return values;
	}

	/**
	 * @return the number of {@link Neuron}s in this {@link Layer}, including bias
	 *         neurons
	 */
	public int size() {
		return neurons.length;
	}

	/**
	 * @return the neuron in this {@link Layer} at a given position
	 */
	public Neuron get(int position) {
		return neurons[position];
	}

	/**
	 * @return true if and only if this {@link Layer} contains a bias {@link Neuron}
	 */
	public boolean containsBias() {
		return bias;
	}

}
