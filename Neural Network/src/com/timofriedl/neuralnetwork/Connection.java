package com.timofriedl.neuralnetwork;

import java.util.Random;

/**
 * Represents a connection of two {@link Neuron}s of a {@link DNN}.
 * 
 * @author Timo Friedl
 */
public class Connection {

	/**
	 * the {@link Neuron}s that are connected
	 */
	private final Neuron from, to;

	/**
	 * the current weight value of this connection
	 */
	private double weight;

	/**
	 * Creates a new connection of two {@link Neuron}s.
	 * 
	 * @param from        the start neuron of this connection
	 * @param to          the end neuron of this connection
	 * @param inputLength the number of neurons in the {@link Layer} of the
	 *                    <code>from</code> neuron
	 */
	public Connection(Neuron from, Neuron to, int inputLength) {
		this.from = from;
		this.to = to;

		weight = new Random().nextGaussian() * Math.sqrt(2.0 / inputLength);
	}

	/**
	 * @return the start {@link Neuron} of this {@link Connection}
	 */
	public Neuron getFrom() {
		return from;
	}

	/**
	 * @return the end {@link Neuron} of this {@link Connection}
	 */
	public Neuron getTo() {
		return to;
	}

	/**
	 * @return the current weight value of this {@link Connection}
	 */
	public double getWeight() {
		return weight;
	}

	/**
	 * @param weight the new weight value of this {@link Connection}
	 */
	public void setWeight(double weight) {
		this.weight = weight;
	}

}
