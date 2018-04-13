<?php

namespace Rubix\Engine\NeuralNetwork\ActivationFunctions;

class PReLU implements ActivationFunction
{
    /**
     * The amount of leakage to allow to pass through when not activated.
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
}
