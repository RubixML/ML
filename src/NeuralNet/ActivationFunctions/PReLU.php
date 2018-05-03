<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

class PReLU implements ActivationFunction
{
    /**
     * The amount of leakage as a ratio of the input value to allow to pass
     * through when not activated.
     *
     * @var float
     */
    protected $leakage;

    /**
     * @param  float  $leakage
     * @return void
     */
    public function __construct(float $leakage = 0.01)
    {
        $this->leakage = $leakage;
    }

    /**
     * Compute the output value.
     *
     * @param  float  $value
     * @return float
     */
    public function compute(float $value) : float
    {
        return $value >= 0.0 ? $value : $this->leakage * $value;
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
        return $computed >= 0.0 ? 1.0 : $this->leakage;
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
