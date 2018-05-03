<?php

namespace Rubix\Engine\NeuralNet;

use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;

class Neuron implements Node
{
    /**
     * The sum of all incoming nerve impulses.
     *
     * @var float
     */
    protected $z;

    /**
     * The precomputed output of the neuron.
     *
     * @var float|null
     */
    protected $precomputed;

    /**
     * The function that determines the the neuron's activation as a function of
     * its inputs.
     *
     * @var \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction
     */
    protected $activationFunction;

    /**
     * The connections made from inbound axons.
     *
     * @var array
     */
    protected $synapses = [
        //
    ];

    /**
     * @param  \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction  $activationFunction
     * @return void
     */
    public function __construct(ActivationFunction $activationFunction)
    {
        $this->z = 0.0;
        $this->precomputed = null;
        $this->activationFunction = $activationFunction;
    }

    /**
     * @return array
     */
    public function synapses() : array
    {
        return $this->synapses;
    }

    /**
     * Return the in degree of the neuron. i.e. the number of incoming connections.
     *
     * @return int
     */
    public function inDegree() : int
    {
        return count($this->synapses);
    }

    /**
     * @return float
     */
    public function z() : float
    {
        return $this->z;
    }

    /**
     * The output signal of the neuron.
     *
     * @return float
     */
    public function output() : float
    {
        if (!isset($this->precomputed)) {
            foreach ($this->synapses as $synapse) {
                $this->z += $synapse->impulse();
            }

            $this->precomputed = $this->activationFunction->compute($this->z);
        }

        return $this->precomputed;
    }

    /**
     * The slope of the gradient with respect to the neuron's output.
     *
     * @return float
     */
    public function slope() : float
    {
        return $this->activationFunction->differentiate($this->z, $this->precomputed);
    }

    /**
     * Connect this neuron to another neuron via a synapse.
     *
     * @param  \Rubix\Engine\NeuralNet\Synapse  $synapse
     * @return \Rubix\Engine\NeuralNet\Synapse
     */
    public function connect(Synapse $synapse) : Synapse
    {
        $this->synapses[] = $synapse;

        return $synapse;
    }

    /**
     * Sever the connection to a given neuron by pruning the synapse.
     *
     * @param  \Rubix\Engine\NeuralNet\Neuron  $neuron
     * @return self
     */
    public function prune(Synapse $synapse) : self
    {
        unset($this->synapses[array_search($synapse)]);

        return $this;
    }

    /**
     * Randomize this neuron's synapse weights.
     *
     * @param  \Rubix\Engine\NeuralNet\Neuron  $neuron
     * @return self
     */
    public function zap() : self
    {
        foreach ($this->synapses as $synapse) {
            list($min, $max) = $this->activationFunction->initialize($this->inDegree());

            $synapse->randomize($min, $max);
        }

        return $this;
    }

    /**
     * Reset the neuron state.
     *
     * @return void
     */
    public function reset() : void
    {
        $this->z = 0.0;
        $this->precomputed = null;
    }
}
