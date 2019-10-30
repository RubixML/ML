<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameters\Parameter;
use InvalidArgumentException;

/**
 * Momentum
 *
 * Momentum adds velocity to each step until exhausted. It does so by accumulating
 * momentum from past updates and adding a factor of the previous velocity to the
 * current step.
 *
 * References:
 * [1] D. E. Rumelhart et al. (1988). Learning representations by back-propagating
 * errors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Momentum implements Optimizer, Adaptive
{
    /**
     * The learning rate that controls the global step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The rate at which the momentum force decays.
     *
     * @var float
     */
    protected $decay;

    /**
     * The parameter cache of velocity matrices.
     *
     * @var \Tensor\Tensor[]
     */
    protected $cache = [
        //
    ];

    /**
     * @param float $rate
     * @param float $decay
     * @throws \InvalidArgumentException
     */
    public function __construct(float $rate = 0.001, float $decay = 0.1)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($decay <= 0. or $decay >= 1.) {
            throw new InvalidArgumentException('Decay must be between'
                . " 0 and 1, $decay given.");
        }

        $this->rate = $rate;
        $this->decay = $decay;
    }

    /**
     * Warm the cache.
     *
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter $param
     */
    public function warm(Parameter $param) : void
    {
        $this->cache[$param->id()] = get_class($param->w())::zeros(...$param->w()->shape());
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameters\Parameter $param
     * @param \Tensor\Tensor $gradient
     * @return \Tensor\Tensor
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor
    {
        $velocity = $this->cache[$param->id()];

        $velocity = $gradient->multiply($this->rate)
            ->add($velocity->multiply(1. - $this->decay));

        $this->cache[$param->id()] = $velocity;

        return $velocity;
    }
}
