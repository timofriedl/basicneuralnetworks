package com.timofriedl.neuralnetwork.training;

/**
 * A Listener for "found new best neural network"-events.
 * 
 * @author Timo Friedl
 */
public interface ProgressListener {

	/**
	 * This method will be called by a training function every time a new best
	 * neural network is found.
	 * 
	 * @param e the {@link ProgressEvent} with information about the training
	 *          status.
	 */
	public void progressChanged(ProgressEvent e);

}
