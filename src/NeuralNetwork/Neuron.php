<?php

namespace Rubix\Engine\NeuralNetwork;

class Neuron
{
    /**
     * The connections made from inbound axons.
     *
     * @var array
     */
    protected $synapses;

    /**
     * @return void
     */
    public function __construct()
    {
        $this->synapses = [];
    }

    /**
     * @return array
     */
    public function synapses() : array
    {
        return $this->synapses;
    }

    /**
     * Connect this neuron to another neuron.
     *
     * @param  \Rubix\Engine\NeuralNetwork\Neuron  $neuron
     * @return \Rubix\Engine\NeuralNetwork\Synapse
     */
    public function connect(self $neuron) : Synapse
    {
        $synapse = new Synapse($neuron);

        $this->synapses[] = $synapse;

        return $synapse;
    }

    /**
     * Prune the connection to a given neuron.
     *
     * @param  \Rubix\Engine\NeuralNetwork\Neuron  $neuron
     * @return self
     */
    public function prune(Synapse $synapse) : self
    {
        unset($this->synapses[array_search($synapse)]);

        return $this;
    }
}
