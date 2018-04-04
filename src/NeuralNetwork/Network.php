<?php

namespace Rubix\Engine\NeuralNetwork;

use Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction;

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
     * Return the input layer.
     *
     * @return array
     */
    public function inputs() : array
    {
        return $this->layers[0];
    }

    /**
     * Return the output layer.
     *
     * @return array
     */
    public function outputs() : array
    {
        return $this->layers[count($this->layers) - 1];
    }

    /**
     * Add the input layer of neurons.
     *
     * @param  int  $inputs
     * @return array
     */
    public function addInputLayer(int $inputs) : array
    {
        $layer = [];

        foreach (range(1, $inputs) as $i) {
            $layer[] = new Input();
        }

        array_push($layer, new Bias());

        $this->layers[] = $layer;

        return $layer;
    }

    /**
     * Add a hidden layer of n neurons using given activation function.
     *
     * @param  int  $n
     * @param  \Rubix\Engine\NeuralNetwork\ActivationsFunctions\ActivationFunction  $activationFunction
     * @return array
     */
    public function addHiddenLayer(int $n, ActivationFunction $activationFunction) : array
    {
        $layer = [];

        foreach (range(1, $n) as $i) {
            $layer[] = new Hidden($activationFunction);
        }

        array_push($layer, new Bias());

        $this->layers[] = $layer;

        return $layer;
    }

    /**
     * Add an output layer of neurons.
     *
     * @param  array  $outcomes
     * @param  \Rubix\Engine\NeuralNetwork\ActivationsFunctions\ActivationFunction  $acctivationFunction
     * @return array
     */
    public function addOutputLayer(array $outcomes, ActivationFunction $activationFunction) : array
    {
        $outcomes = array_unique($outcomes);
        $layer = [];

        foreach ($outcomes as $outcome) {
            $layer[] = new Output($outcome, $activationFunction);
        }

        $this->layers[] = $layer;

        return $layer;
    }

    /**
     * Fully connect layer a to layer b.
     *
     * @param  array  $a
     * @param  array  $b
     * @return self
     */
    public function connectLayers(array $a, array $b) : self
    {
        foreach ($a as $next) {
            if ($next instanceof Neuron) {
                foreach ($b as $current) {
                    $synapse = Synapse::init($current);

                    $next->connect($synapse);
                }
            }
        }

        return $this;
    }

    /**
     * Reset the z values for all neurons in the network.
     *
     * @return void
     */
    public function reset() : void
    {
        foreach (range(1, count($this->layers) - 1) as $i) {
            foreach ($this->layers[$i] as $neuron) {
                if ($neuron instanceof Hidden) {
                    $neuron->reset();
                }
            }
        }
    }
}
