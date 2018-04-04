<?php

namespace Rubix\Engine\NeuralNetwork;

use Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction;

class Hidden extends Neuron
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
     * @var float
     */
    protected $precomputed;

    /**
     * The function that determines the magnitude of this neurons output.
     *
     * @var \Rubix\Engine\ActivationFunctions\ActivationFunction
     */
    protected $activationFunction;

    /**
     * @param  \Rubix\Engine\NeuralNetwork\ActivationFunctions\ActivationFunction  $activationFunction
     * @return void
     */
    public function __construct(ActivationFunction $activationFunction)
    {
        parent::__construct();

        $this->activationFunction = $activationFunction;

        $this->reset();
    }

    /**
     * The output signal of the hidden neuron.
     *
     * @return float
     */
    public function output() : float
    {
        if (!isset($this->precomputed)) {
            $this->z = 0.0;

            foreach ($this->synapses as $synapse) {
                $this->z += $synapse->impulse();
            }

            $this->precomputed = $this->activationFunction->compute($this->z);
        }

        return $this->precomputed;
    }

    /**
     * The slope of the error at the given output.
     *
     * @return float
     */
    public function error() : float
    {
        return $this->activationFunction->differentiate($this->z, $this->output());
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
