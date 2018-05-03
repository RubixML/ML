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
            if ($layer instanceof Hidden) {
                $this->layers[] = $layer;
            }
        }

        $this->layers[] = $output;

        for ($i = $this->depth() - 1; $i > 0; $i--) {
            $this->connectLayers($this->layers[$i], $this->layers[$i - 1]);
        }
    }

    /**
     * Return the input layer.
     *
     * @return array
     */
    public function inputs() : Input
    {
        return $this->layers[0];
    }

    /**
     * Return an array of hidden layers indexed left to right.
     *
     * @return array
     */
    public function hiddens() : array
    {
        return array_slice($this->layers, 1, count($this->layers) - 2, true);
    }

    /**
     * Return the output layer.
     *
     * @return array
     */
    public function outputs() : Output
    {
        return $this->layers[count($this->layers) - 1];
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
     * Feed a sample through the network and return the output activation of each neuron.
     *
     * @param  array  $sample
     * @throws \RuntimeException
     * @return array
     */
    public function feed(array $sample) : array
    {
        if (count($sample) !== $this->inputs()->count() - 1) {
            throw new RuntimeException('The ratio of feature columns to input neurons is unequal, '
                . (string) count($sample) . ' found, ' . (string) ($this->inputs()->count() - 1) . ' needed.');
        }

        $this->reset();

        $this->inputs()->prime($sample);

        return $this->outputs()->fire();
    }

    /**
     * Return an array of paramters. i.e. the weights of each synapse in the network.
     *
     * @return array
     */
    public function readParameters() : array
    {
        $parameters = [];

        foreach (array_slice($this->layers, 1) as $i => $layer) {
            foreach ($layer as $j => $neuron) {
                if ($neuron instanceof Neuron) {
                    foreach ($neuron->synapses() as $k => $synapse) {
                        $parameters[$i][$j][$k] = $synapse->weight();
                    }
                }
            }
        }

        return $parameters;
    }

    /**
     * Restore the network parameters from an array of weights indexed by layer
     * then neuron then finally synapse.
     *
     * @param  array  $parameters
     * @return void
     */
    public function restoreParameters(array $parameters) : void
    {
        foreach (array_slice($this->layers, 1) as $i => $layer) {
            foreach ($layer as $j => $neuron) {
                if ($neuron instanceof Neuron) {
                    foreach ($neuron->synapses() as $k => $synapse) {
                        $synapse->setWeight($parameters[$i][$j][$k]);
                    }
                }
            }
        }
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
                if ($neuron instanceof Neuron) {
                    $neuron->zap();
                }
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
            if ($next instanceof Neuron) {
                foreach ($b as $current) {
                    $next->connect(new Synapse($current));
                }
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
                if ($neuron instanceof Neuron) {
                    $neuron->reset();
                }
            }
        }
    }
}
