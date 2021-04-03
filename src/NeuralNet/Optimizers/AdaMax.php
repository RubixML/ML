<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;

use function get_class;

use const Rubix\ML\EPSILON;

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
 */
class AdaMax extends Adam
{
    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @internal
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Tensor\Tensor<int|float|array> $gradient
     * @return \Tensor\Tensor<int|float|array>
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor
    {
        [$velocity, $norm] = $this->cache[$param->id()];

        $velocity = $velocity->multiply($this->beta1)
            ->add($gradient->multiply($this->momentumDecay));

        $class = get_class($param->param());

        $norm = $class::maximum($norm->multiply($this->beta2), $gradient->abs());

        $this->cache[$param->id()] = [$velocity, $norm];

        if ($this->t < self::WARM_UP_STEPS) {
            ++$this->t;

            $rate = $this->rate / (1.0 - $this->beta1 ** $this->t);
        } else {
            $rate = $this->rate;
        }

        return $velocity->divide($norm->clipLower(EPSILON))->multiply($rate);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "AdaMax (rate: {$this->rate}, momentum_decay: {$this->momentumDecay},"
            . " norm_decay: {$this->normDecay})";
    }
}
