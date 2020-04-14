package com.timofriedl.neuralnetwork.training;

import java.util.ArrayList;
import java.util.Collections;

import com.timofriedl.neuralnetwork.DNN;

/**
 * An evolutionary algorithm to train deep neural networks ({@link DNN}).
 * 
 * @author Timo Friedl
 */
public abstract class EvolutionaryTraining {

	/**
	 * Trains this {@link DNN} using an evolutionary algorithm.
	 * 
	 * @param first            the first neural network of this population
	 * @param inputs           the array of input values to feed the netrowk
	 * @param ideals           the array of ideal values to use for comparison
	 * @param populationSize   the number of individuals in the population
	 * @param killRate         the part of the population that will be killed each
	 *                         epoch
	 * @param errorAim         the aimed error value
	 * @param maxEpochs        the maximal number of epochs to train before stopping
	 *                         this function
	 * @param progressListener (optional) the {@link ProgressListener} that informs
	 *                         you when a new best neural network was found.
	 * @return the best DNN
	 */
	public static DNN train(DNN first, double[][] inputs, double[][] ideals, int populationSize, double killRate,
			double errorAim, int maxEpochs, ProgressListener progressListener) {
		// Init population
		ArrayList<DNN> population = initPopulation(first, populationSize);

		// Init best network and error
		DNN best = null;
		double error = netError(first, inputs, ideals);

		// Train
		for (int epoch = 0; epoch < maxEpochs; epoch++) {
			sortNets(population, inputs, ideals);

			// Check for new best network
			if (population.get(0) != best) {
				best = population.get(0);
				error = netError(best, inputs, ideals);

				if (progressListener != null)
					progressListener.progressChanged(new ProgressEvent(best, error, epoch));

				if (error <= errorAim)
					return best;
			}

			// Kill bad
			for (int i = 0; i < killRate * populationSize; i++)
				killBad(population);

			// Mutate good
			while (population.size() < populationSize)
				addMutation(population);
		}

		// Return the best network
		sortNets(population, inputs, ideals);
		return population.get(0);
	}

	/**
	 * Initializes a population with a given first individual and a given population
	 * size.
	 * 
	 * @param first          the first individual
	 * @param populationSize the number of individuals in the population
	 * @return the population
	 */
	private static ArrayList<DNN> initPopulation(DNN first, int populationSize) {
		ArrayList<DNN> population = new ArrayList<>();
		population.add(first);

		while (population.size() < populationSize)
			population.add(new DNN(first.getLayerSizes()));

		return population;
	}

	/**
	 * Trains this {@link DNN} using an evolutionary algorithm.
	 * 
	 * @param first          the first neural network of this population
	 * @param inputs         the array of input values to feed the netrowk
	 * @param ideals         the array of ideal values to use for comparison
	 * @param populationSize the number of individuals in the population
	 * @param killRate       the part of the population that will be killed each
	 *                       epoch
	 * @param errorAim       the aimed error value
	 * @param maxEpochs      the maximal number of epochs to train before stopping
	 *                       this function
	 * @return the best DNN
	 */
	public static DNN train(DNN first, double[][] inputs, double[][] ideals, int populationSize, double killRate,
			double errorAim, int maxEpochs) {
		return train(first, inputs, ideals, populationSize, killRate, errorAim, maxEpochs, null);
	}

	/**
	 * Sorts the population by error ascending.
	 * 
	 * @param population the population to sort
	 * @param inputs     the array of input training data
	 * @param ideals     the array of expected output training data
	 */
	private static void sortNets(ArrayList<DNN> population, double[][] inputs, double[][] ideals) {
		Collections.sort(population,
				(i1, i2) -> Double.compare(netError(i1, inputs, ideals), netError(i2, inputs, ideals)));
	}

	/**
	 * Calculates the total error of a given neural network, regarding given input
	 * and expected output values.
	 * 
	 * @param net    the network to test
	 * @param inputs the array of input values to feed the network
	 * @param ideals the array of expected output values
	 * @return the total error
	 */
	public static double netError(DNN net, double[][] inputs, double[][] ideals) {
		double error = 0.0;

		for (int i = 0; i < inputs.length; i++) {
			final double[] input = inputs[i];
			final double[] output = net.feedForward(input);
			final double[] ideal = ideals[i];

			for (int j = 0; j < output.length; j++)
				error += Math.pow(ideal[j] - output[j], 2.0);
		}

		return error;
	}

	/**
	 * Kills a (probably) bad individual of a given population. The population must
	 * be sorted by error ascending.
	 * 
	 * @param population the population to decimate
	 */
	private static void killBad(ArrayList<DNN> population) {
		int index = population.size() - 1;

		while (Math.random() < 0.5)
			index--;

		if (index < 0)
			index = population.size() - 1;

		population.remove(index);
	}

	/**
	 * Adds a mutated individual of a (probably) good individual. The population
	 * must be sorted by error ascending.
	 * 
	 * @param population the population to increase
	 */
	private static void addMutation(ArrayList<DNN> population) {
		int index = 0;

		while (Math.random() < 0.5)
			index++;

		if (index >= population.size())
			index = 0;

		population.add(population.get(index).mutate(4.0 * Math.random() / population.get(index).getWeightCount()));
	}

}
