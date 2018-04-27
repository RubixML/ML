<?php

namespace Rubix\Engine\NeuralNetwork;

use Rubix\Engine\NeuralNetwork\ActivationFunctions\Sigmoid;
use Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction;
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
     * @param  int  $inputs
     * @param  array  $hidden
     * @param  array  $outcomes
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $inputs, array $hidden, array $outcomes)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException('The number of inputs must be greater than 0.');
        }

        foreach ($hidden as &$layer) {
            if (!is_array($layer)) {
                $layer = [$layer];
            }

            if (!is_int($layer[0]) || $layer[0] < 1) {
                throw new InvalidArgumentException('The size parameter of a hidden layer must be an integer greater than 0.');
            }

            if (isset($layer[1])) {
                if (!$layer[1] instanceof ActivationFunction) {
                    throw new InvalidArgumentException('The second hidden layer parameter must be an instance of an ActivationFunction.');
                }
            }
        }

        if (count($outcomes) < 2) {
            throw new InvalidArgumentException('The number of unique outcomes must be greater than 1.');
        }

        $this->layers[] = new InputLayer($inputs);

        foreach ($hidden as $layer) {
            $this->layers[] = new HiddenLayer($layer[0], $layer[1] ?? new Sigmoid());
        }

        $this->layers[] = new OutputLayer($outcomes, new Sigmoid());

        for ($layer = count($this->layers) - 1; $layer > 0; $layer--) {
            $this->connectLayers($this->layers[$layer], $this->layers[$layer - 1]);
        }
    }

    /**
     * Return the input layer.
     *
     * @return array
     */
    public function inputs() : InputLayer
    {
        return $this->layers[0];
    }

    /**
     * Return the output layer.
     *
     * @return array
     */
    public function outputs() : OutputLayer
    {
        return $this->layers[count($this->layers) - 1];
    }

    /**
     * The depth of the network. i.e. the number of hidden layers.
     *
     * @return int
     */
    public function depth() : int
    {
        return count($this->layers) - 2;
    }

    /**
     * Feed a sample through the network and calculate the output of each neuron.
     *
     * @param  array  $sample
     * @throws \RuntimeException
     * @return void
     */
    public function feed(array $sample) : void
    {
        if (count($sample) !== count($this->layers[0]) - 1) {
            throw new RuntimeException('The number of feature columns must equal the number of input neurons.');
        }

        $this->reset();

        $column = 0;

        foreach ($this->inputs() as $input) {
            if ($input instanceof Input) {
                $input->prime($sample[$column++]);
            }
        }

        foreach ($this->outputs() as $output) {
            $output->fire();
        }
    }

    /**
     * Return an array of paramters. i.e. the weights of each synapse in the network.
     *
     * @return array
     */
    public function readParameters() : array
    {
        $paramters = [];

        for ($layer = 1; $layer < count($this->layers); $layer++) {
            foreach ($this->layers[$layer] as $i => $neuron) {
                foreach ($neuron->synapses() as $j => $synapse) {
                    $parameters[$layer][$i][$j] = $synapse->weight();
                }
            }
        }

        return $parameters;
    }

    /**
     * Return the output vector of the network.
     *
     * @return array
     */
    public function readOutput() : array
    {
        $activations = [];

        foreach ($this->outputs() as $neuron) {
            $activations[] = $neuron->output();
        }

        return $activations;
    }

    /**
     * Randomize the weights for all synapses in the network.
     *
     * @return void
     */
    public function randomizeWeights() : void
    {
        for ($layer = 1; $layer < count($this->layers); $layer++) {
            foreach ($this->layers[$layer] as $neuron) {
                $neuron->zap();
            }
        }
    }

    /**
     * Fully connect layer a to layer b.
     *
     * @param  array  $a
     * @param  array  $b
     * @return self
     */
    public function connectLayers(Layer $a, Layer $b) : self
    {
        foreach ($a as $next) {
            foreach ($b as $current) {
                $next->connect(new Synapse($current));
            }
        }

        return $this;
    }

    /**
     * Reset the computed values for all neurons in the network.
     *
     * @return void
     */
    public function reset() : void
    {
        for ($layer = 1; $layer < count($this->layers); $layer++) {
            foreach ($this->layers[$layer] as $neuron) {
                if ($neuron instanceof Hidden) {
                    $neuron->reset();
                }
            }
        }
    }
}
