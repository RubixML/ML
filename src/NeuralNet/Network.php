<?php

namespace Rubix\Engine\NeuralNet;

use Rubix\Engine\NeuralNet\Layers\Layer;
use Rubix\Engine\NeuralNet\Layers\Input;
use Rubix\Engine\NeuralNet\Layers\Hidden;
use Rubix\Engine\NeuralNet\Layers\Output;
use InvalidArgumentException;
use RuntimeException;

class Network
{
    /**
     * The layers of the network.
     *
     * @var array
     */
    protected $layers = [
        //
    ];

    /**
     * @param  \Rubix\Engine\NeuralNet\Layers\Input  $input
     * @param  array  $hidden
     * @param  \Rubix\Engine\NeuralNet\Layers\Output  $output
     * @return void
     */
    public function __construct(Input $input, array $hidden, Output $output)
    {
        $this->layers[] = $input;

        foreach ($hidden as $layer) {
            if (!$layer instanceof Hidden) {
                throw new InvalidArgumentException('Hidden layers must only extend the Hidden class, '
                    . get_class($layer) . ' found.');
            }

            $this->layers[] = $layer;
        }

        $this->layers[] = $output;
    }

    /**
     * Return the input layer.
     *
     * @return array
     */
    public function input() : Input
    {
        return $this->layers[0];
    }

    /**
     * Return an array of hidden layers indexed left to right.
     *
     * @return array
     */
    public function hidden() : array
    {
        return array_slice($this->layers, 1, count($this->layers) - 2, true);
    }

    /**
     * Return the output layer.
     *
     * @return array
     */
    public function output() : Output
    {
        return $this->layers[count($this->layers) - 1];
    }

    /**
     * The parametric layers of the network. i.e. the layers that have weights.
     *
     * @return array
     */
    public function parametric() : array
    {
        return array_slice($this->layers, 1, count($this->layers), true);
    }

    /**
     * The depth of the network. i.e. the number of layers.
     *
     * @return int
     */
    public function depth() : int
    {
        return count($this->layers);
    }

    /**
     * Feed a sample through the network and return the output activations
     * of each output neuron.
     *
     * @param  array  $sample
     * @return array
     */
    public function feed(array $sample) : array
    {
        $activations = $sample;

        foreach ($this->layers as $layer) {
            $activations = $layer->forward($activations);
        }

        return $activations;
    }

    /**
     * Backpropagate the error determined by the given outcome and return the
     * gradients at each layer.
     *
     * @param  mixed  $outcome
     * @return array
     */
    public function backpropagate($outcome) : array
    {
        $layers = [];

        foreach (array_reverse($this->parametric(), true) as $i => $layer) {
            if ($layer instanceof Output) {
                list($errors, $gradients) = $layer->back($outcome);
            } else {
                list($errors, $gradients) = $layer->back($previousWeights, $previousErrors);
            }

            $previousWeights = $layer->parameters();
            $previousErrors = $errors;

            $layers[$i] = $gradients;
        }

        return $layers;
    }

    /**
     * Randomize the weights for all synapses in the network.
     *
     * @return void
     */
    public function initialize() : void
    {
        for ($i = 1; $i < count($this->layers); $i++) {
            $current = $this->layers[$i];
            $previous = $this->layers[$i - 1];

            $current->initialize($previous);
        }
    }

    /**
     * Return an array of paramters. i.e. the weights of each synapse in the network.
     *
     * @return array
     */
    public function readParameters() : array
    {
        $layers = [];

        foreach ($this->parametric() as $i => $layer) {
            $layers[$i] = $layer->parameters();
        }

        return $layers;
    }

    /**
     * Restore the network parameters from an array of weights indexed by layer
     * then neuron then finally synapse weight.
     *
     * @param  array  $parameters
     * @return void
     */
    public function restoreParameters(array $parameters) : void
    {
        foreach ($this->parametric() as $i => $layer) {
            $layer->setParameters($parameters[$i]);
        }
    }
}
