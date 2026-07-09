<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use NDArray;
use NumPower;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Specifications\SpecificationChain;

use const Rubix\ML\EPSILON;
use const PHP_FLOAT_MAX;

/**
 * AdaMax
 *
 * A version of Adam that replaces the RMS property with the infinity norm of the gradients.
 *
 * References:
 * [1] D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
class AdaMax extends Adam
{
    /**
     * @param float $rate
     * @param float $momentumDecay
     * @param float $normDecay
     */
    public function __construct(float $rate = 0.001, float $momentumDecay = 0.1, float $normDecay = 0.001)
    {
        parent::__construct($rate, $momentumDecay, $normDecay);

        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * AdaMax update (element-wise):
     *   v_t = v_{t-1} + β1 · (g_t − v_{t-1})
     *   u_t = max(β2 · u_{t-1}, |g_t|)
     *   Δθ_t = η · v_t / max(u_t, ε)
     *
     * @internal
     *
     * @param Parameter $param
     * @param NDArray $gradient
     * @return NDArray
     */
    public function step(Parameter $param, NDArray $gradient) : NDArray
    {
        [$velocity, $norm] = $this->cache[$param->id()];

        $vHat = NumPower::multiply(
            NumPower::subtract($gradient, $velocity),
            $this->momentumDecay
        );

        $velocity = NumPower::add($velocity, $vHat);

        // Infinity norm accumulator
        $norm = NumPower::multiply($norm, 1.0 - $this->normDecay);
        $absGrad = NumPower::abs($gradient);
        $norm = NumPower::maximum($norm, $absGrad);

        $this->cache[$param->id()] = [$velocity, $norm];

        $norm = NumPower::clip($norm, EPSILON, PHP_FLOAT_MAX);

        return NumPower::multiply(
            NumPower::divide($velocity, $norm),
            $this->rate
        );
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "AdaMax (rate: {$this->rate}, momentum decay: {$this->momentumDecay},"
            . " norm decay: {$this->normDecay})";
    }
}
