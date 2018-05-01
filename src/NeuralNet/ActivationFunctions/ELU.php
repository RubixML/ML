<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

class ELU implements ActivationFunction
{
    /**
     * Alpha defines at which negative value the ELU saturates. i.e. a = 1.0 means
     * that the value will be lower than -1.0.
     *
     * @var float
     */
    protected $alpha;

    /**
     * @param  float  $leakage
     * @return void
     */
    public function __construct(float $alpha = 1.0)
    {
        $this->alpha = $alpha;
    }

    /**
     * Compute the output value.
     *
     * @param  float  $value
     * @return float
     */
    public function compute(float $value) : float
    {
        return $value >= 0.0 ? $value : $this->alpha * (exp($value) - 1);
    }

    /**
     * Calculate the partial derivative with respect to the computed output.
     *
     * @param  float  $value
     * @param  float  $computed
     * @return float
     */
    public function differentiate(float $value, float $computed) : float
    {
        return $computed >= 0.0 ? 1.0 : $computed + $this->alpha;
    }

    /**
     * Generate an initial synapse weight range based on n, the number of inputs
     * to a particular neuron.
     *
     * @param  \Rubix\Engine\NeuralNet\Synapse  $synapse
     * @param  int  $n
     * @return array
     */
    public function initialize(int $n) : array
    {
        $r = pow(6 / $n, 1 / self::ROOT_2);

        return [-$r, $r];
    }
}
