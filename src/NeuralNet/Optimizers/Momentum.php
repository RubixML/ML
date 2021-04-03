<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function get_class;

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
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(float $rate = 0.001, float $decay = 0.1)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($decay <= 0.0 or $decay >= 1.0) {
            throw new InvalidArgumentException('Decay must be between'
                . " 0 and 1, $decay given.");
        }

        $this->rate = $rate;
        $this->decay = $decay;
    }

    /**
     * Warm the cache.
     *
     * @internal
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     */
    public function warm(Parameter $param) : void
    {
        $class = get_class($param->param());

        $this->cache[$param->id()] = $class::zeros(...$param->param()->shape());
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @internal
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Tensor\Tensor<int|float|array> $gradient
     * @return \Tensor\Tensor<int|float|array>
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor
    {
        $velocity = $this->cache[$param->id()];

        $velocity = $gradient->multiply($this->rate)
            ->add($velocity->multiply(1.0 - $this->decay));

        $this->cache[$param->id()] = $velocity;

        return $velocity;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Momentum (rate: {$this->rate}, decay: {$this->decay})";
    }
}
