<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Output;

interface Network
{
    /**
     * Return all the layers in the network.
     *
     * @return array
     */
    public function layers() : array;

    /**
     * Return the input layer.
     *
     * @return \Rubix\ML\NeuralNet\Layers\Input
     */
    public function input() : Input;

    /**
     * Return an array of hidden layers indexed left to right.
     *
     * @return array
     */
    public function hidden() : array;

    /**
     * Return the output layer.
     *
     * @return \Rubix\ML\NeuralNet\Layers\Output
     */
    public function output() : Output;

    /**
     * The parametric layers of the network. i.e. the layers that have weights.
     *
     * @return array
     */
    public function parametric() : array;

    /**
     * The depth of the network. i.e. the number of parametric layers.
     *
     * @return int
     */
    public function depth() : int;

    /**
     * Feed a sample through the network and return the output activations
     * of each output neuron.
     *
     * @param  array  $samples
     * @return self
     */
    public function feed(array $samples);

    /**
     * Return the activations of the neurons at the output layer.
     *
     * @param  array  $samples
     * @return array
     */
    public function infer(array $samples) : array;

    /**
     * Backpropagate the error determined by the previous layer and take a step
     * in the direction of the steepest descent.
     *
     * @param  array  $labels
     * @return float
     */
    public function backpropagate(array $labels) : float;

    /**
     * Restore the network parameters from a snapshot.
     *
     * @param  \Rubix\ML\NeuralNet\Snapshot  $snapshot
     * @return void
     */
    public function restore(Snapshot $snapshot) : void;
}
