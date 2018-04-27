<?php

namespace Rubix\Engine\NeuralNet;

use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;

class Hidden extends Neuron
{
    /**
     * The sum of all incoming nerve impulses.
     *
     * @var float
     */
    protected $z = 0.0;

    /**
     * The precomputed output of the neuron.
     *
     * @var float|null
     */
    protected $precomputed = null;

    /**
     * The function that determines the magnitude of this neurons output.
     *
     * @var \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction
     */
    protected $activationFunction;

    /**
     * @param  \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction  $activationFunction
     * @return void
     */
    public function __construct(ActivationFunction $activationFunction)
    {
        $this->activationFunction = $activationFunction;
    }

    /**
     * The output signal of the hidden neuron.
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
     * The slope of the error at the given output.
     *
     * @return float
     */
    public function derivative() : float
    {
        return $this->activationFunction->differentiate($this->z, $this->precomputed);
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
