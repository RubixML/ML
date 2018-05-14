<?php

namespace Rubix\Engine\NeuralNet\ActivationFunctions;

use InvalidArgumentException;

class ELU implements ActivationFunction
{
    /**
     * At which negative value the ELU will saturate. i.e. alpha = 1.0 means
     * that the leakage will never be more than -1.0.
     *
     * @var float
     */
    protected $alpha;

    /**
     * @param  float  $alpha
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $alpha = 1.0)
    {
        if ($alpha < 0) {
            throw new InvalidArgumentException('Alpha parameter must be a positive value.');
        }

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
     * @param  int  $inDegree
     * @return float
     */
    public function initialize(int $inDegree) : float
    {
        $r = pow(6 / $inDegree, 1 / self::ROOT_2);

        $scale = pow(10, 10);

        return random_int(-$r * $scale, $r * $scale) / $scale;
    }
}
