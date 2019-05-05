<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameters\Parameter;

use const Rubix\ML\EPSILON;

/**
 * AdaMax
 *
 * A version of Adam that replaces the RMS property with the infinity norm of
 * the gradients.
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
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter $param
     * @param \Rubix\Tensor\Tensor $gradient
     */
    public function step(Parameter $param, Tensor $gradient) : void
    {
        [$velocity, $norm] = $this->cache[$param->id()];

        $velocity = $velocity->multiply($this->beta1)
            ->add($gradient->multiply($this->momentumDecay));

        $tensor = get_class($param->w());

        $norm = $tensor::maximum($norm->multiply($this->beta2), $gradient->abs());

        $this->cache[$param->id()] = [$velocity, $norm];

        if ($this->t < self::WARM_UP_STEPS) {
            $this->t++;

            $rate = $this->rate / (1. - $this->beta1 ** $this->t);
        } else {
            $rate = $this->rate;
        }

        $step = $velocity->divide($norm->clipLower(EPSILON))
            ->multiply($rate);

        $param->update($step);
    }
}
