package com.timofriedl.neuralnetwork.visual;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;

import com.timofriedl.neuralnetwork.Connection;
import com.timofriedl.neuralnetwork.DNN;
import com.timofriedl.neuralnetwork.Layer;
import com.timofriedl.neuralnetwork.Neuron;

/**
 * A {@link Graphics2D} visualizer for {@link DNN}s.
 * 
 * @author Timo Friedl
 */
public abstract class DNNVisualizer {

	/**
	 * Renders a given {@link DNN}.
	 * 
	 * I guess I wasn't that motivated when writing this method...
	 * 
	 * @param g      the {@link Graphics2D} instance to draw on
	 * @param net    the network to render
	 * @param x      the x position of the upper left corner of the image
	 * @param y      the y position of the upper left corner of the image
	 * @param width  the width of the image
	 * @param height the height of the image
	 */
	public static void drawDNN(Graphics2D g, DNN net, double x, double y, double width, double height) {
		g.setStroke(new BasicStroke(1f));

		int maxLayerSize = 0;
		for (Layer l : net.getLayers()) {
			final int size = l.size();

			if (size > maxLayerSize)
				maxLayerSize = size;
		}

		double minWeight = Double.MAX_VALUE, maxWeigth = -Double.MAX_VALUE;
		for (Layer l : net.getLayers())
			for (int n = 0; n < l.size(); n++) {
				final Neuron nr = l.get(n);
				for (Connection c : nr.getIncomingConnections()) {
					if (c.getWeight() < minWeight)
						minWeight = c.getWeight();
					if (c.getWeight() > maxWeigth)
						maxWeigth = c.getWeight();
				}
			}

		final double circleSize = height / (maxLayerSize * 2 - 1);
		final double xDis = (width - circleSize) / (net.getLayers().length - 1);

		for (int l = 0; l < net.getLayers().length; l++) {
			final Layer layer = net.getLayers()[l];

			final double yMin = 0.5 * (height - circleSize * (layer.size() * 2 - 1));
			final double yMinL = l == 0 ? 0.0 : 0.5 * (height - circleSize * (net.getLayers()[l - 1].size() * 2 - 1));

			for (int n = 0; n < layer.size(); n++) {
				final Neuron neuron = layer.get(n);

				final double cx = x + xDis * l;
				final double cy = y + yMin + n * circleSize * 2.0;

				for (int cn = 0; cn < neuron.getIncomingConnections().length; cn++) {
					final Connection c = neuron.getIncomingConnections()[cn];

					double p = (c.getWeight() - minWeight) / (maxWeigth - minWeight);
					g.setColor(new Color(p > 0.5 ? 0xFF : (int) (0xFF * 2.0 * p),
							p < 0.5 ? 0xFF : (int) (0xFF * (2.0 - 2.0 * p)), 0x00));
					g.draw(new Line2D.Double(cx + circleSize / 2.0, cy + circleSize / 2.0, cx - xDis + circleSize / 2.0,
							y + yMinL + cn * circleSize * 2.0 + circleSize / 2.0));
				}

				g.setColor(Color.WHITE);
				g.draw(new Ellipse2D.Double(cx, cy, circleSize, circleSize));
			}
		}
	}

}
