package com.timofriedl.neuralnetwork.training;

import com.timofriedl.neuralnetwork.DNN;

/**
 * Represents an event in neural network training.
 * 
 * @author Timo Friedl
 */
public class ProgressEvent {

	/**
	 * the new best {@link DNN} to refer to
	 */
	private final DNN net;

	/**
	 * the error of the neural network
	 */
	private final double error;

	/**
	 * the current epoch of the training
	 */
	private final int epoch;

	/**
	 * Creates a new progress event.
	 * 
	 * @param net   the neural network to refer to
	 * @param error the error value of the network
	 */
	public ProgressEvent(DNN net, double error, int epoch) {
		this.net = net;
		this.error = error;
		this.epoch = epoch;
	}

	/**
	 * @return the new best neural network
	 */
	public DNN getNet() {
		return net;
	}

	/**
	 * @return the error of the neural network
	 */
	public double getError() {
		return error;
	}

	/**
	 * @return the current epoch of the training
	 */
	public int getEpoch() {
		return epoch;
	}

}
