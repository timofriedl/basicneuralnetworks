package com.timofriedl.neuralnetwork;

import java.util.ArrayList;

/**
 * Represents a deep neural network consisting of input, output, and hidden
 * layers.
 * 
 * @author Timo Friedl
 */
public class DNN {

	/**
	 * the layers of this neural network
	 */
	private final Layer[] layers;

	/**
	 * the number of neurons in each layer, excluding bias neurons
	 */
	private final int[] layerSizes;

	/**
	 * Creates a new deep neural network with given layer sizes. First layer size is
	 * the number of input neurons, last layer size the number of output neurons.
	 * 
	 * @param layerSizes the number of neurons in each layer, excluding bias neurons
	 * @throws IllegalArgumentException if there are less than two layer sizes
	 */
	public DNN(int... layerSizes) {
		this.layerSizes = layerSizes;

		if (layerSizes.length < 2)
			throw new IllegalArgumentException("A DNN must have at least two layers.");

		layers = new Layer[layerSizes.length];

		for (int i = 0; i < layers.length; i++)
			layers[i] = new Layer(layerSizes[i], i != layers.length - 1, i == 0 ? null : layers[i - 1]);
	}

	/**
	 * Processes a given set of data and returns the computed network result.
	 * 
	 * @param inputs the input values to process
	 * @return the output values of the {@link DNN}
	 * @throws IllegalArgumentException if the number of input values is invalid
	 */
	public double[] feedForward(double[] inputs) {
		final int expectedLength = layers[0].size() - 1;

		if (inputs.length != expectedLength)
			throw new IllegalArgumentException("Expected " + expectedLength + " inputs but got " + inputs.length + ".");

		for (int i = 0; i < inputs.length; i++)
			layers[0].get(i).setValue(inputs[i]);

		for (int i = 1; i < layers.length; i++)
			layers[i].update();

		return layers[layers.length - 1].values();
	}

	/**
	 * Adjusts the weights of this network to modify its behaviour and returns the
	 * result as a new network.
	 * 
	 * @param weightMutationChance the chance for every individual weight to be
	 *                             modified at all, in range [0,1]
	 * @return the mutated network as a new {@link DNN} instance
	 */
	public DNN mutate(double weightMutationChance) {
		final ArrayList<Double> weights = flatWeights();

		for (int i = 0; i < weights.size(); i++)
			if (Math.random() < weightMutationChance)
				if (Math.random() < 0.5) {
					if (Math.random() < 0.5)
						weights.set(i, Math.random() * 2.0 - 1.0);
					else
						weights.set(i, -weights.get(i));
				} else {
					if (Math.random() < 0.5)
						weights.set(i, weights.get(i) * (Math.random() * 2.0 - 1.0));
					else
						weights.set(i, weights.get(i) + (Math.random() - 0.5));
				}

		final DNN clone = new DNN(layerSizes);
		clone.putFlatWeights(weights);

		return equals(clone) ? mutate(weightMutationChance) : clone;
	}

	/**
	 * Collects the {@link Connection} weights of this {@link DNN} to a
	 * one-dimensional {@link ArrayList}.
	 * 
	 * @return the connection weights
	 */
	public ArrayList<Double> flatWeights() {
		final ArrayList<Double> weights = new ArrayList<>();

		for (int i = 0; i < layers.length; i++) {
			final Layer layer = layers[i];

			for (int j = 0; j < layer.size(); j++) {
				final Neuron neuron = layer.get(j);

				for (int k = 0; k < neuron.getIncomingConnections().length; k++)
					weights.add(neuron.getIncomingConnections()[k].getWeight());
			}
		}

		return weights;
	}

	/**
	 * Sets the weights of this networks {@link Connection}s to given values.
	 * 
	 * @param weights the new weight values
	 */
	public void putFlatWeights(ArrayList<Double> weights) {
		int pos = 0;

		for (int i = 0; i < layers.length; i++) {
			final Layer layer = layers[i];

			for (int j = 0; j < layer.size(); j++) {
				final Neuron neuron = layer.get(j);

				for (int k = 0; k < neuron.getIncomingConnections().length; k++)
					neuron.getIncomingConnections()[k].setWeight(weights.get(pos++));
			}
		}

		if (pos < weights.size())
			throw new IllegalArgumentException("Weight values must match the weight count in this network.");
	}

	@Override
	public boolean equals(Object o) {
		if (o == null || !(o instanceof DNN))
			return false;

		final DNN net = (DNN) o;

		return net.layers.equals(layers) && net.flatWeights().equals(flatWeights());
	}

	/**
	 * @return the layers of this {@link DNN}
	 */
	public Layer[] getLayers() {
		return layers;
	}
}