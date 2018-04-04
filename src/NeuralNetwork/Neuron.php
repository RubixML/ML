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
     * Connect this neuron to another neuron via a synapse.
     *
     * @param  \Rubix\Engine\NeuralNetwork\Synapse  $synapse
     * @return \Rubix\Engine\NeuralNetwork\Synapse
     */
    public function connect(Synapse $synapse) : Synapse
    {
        $this->synapses[] = $synapse;

        return $synapse;
    }

    /**
     * Sever the connection to a given neuron by pruning the synapse.
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
