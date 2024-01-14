<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Tensor\Tensor;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function get_class;

use const Rubix\ML\EPSILON;

/**
 * AdaGrad
 *
 * Short for Adaptive Gradient, the AdaGrad Optimizer speeds up the learning of
 * parameters that do not change often and slows down the learning of parameters
 * that do enjoy heavy activity.
 *
 * References:
 * [1] J. Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning
 * and Stochastic Optimization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AdaGrad implements Optimizer, Adaptive
{
    /**
     * The learning rate that controls the global step size.
     *
     * @var float
     */
    protected float $rate;

    /**
     * The cache of sum of squared gradients.
     *
     * @var \Tensor\Tensor[]
     */
    protected array $cache = [
        //
    ];

    /**
     * @param float $rate
     * @throws InvalidArgumentException
     */
    public function __construct(float $rate = 0.01)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must be'
                . " greater than 0, $rate given.");
        }

        $this->rate = $rate;
    }

    /**
     * Warm the parameter cache.
     *
     * @internal
     *
     * @param Parameter $param
     * @throws RuntimeException
     */
    public function warm(Parameter $param) : void
    {
        $class = get_class($param->param());

        if ($class === false) {
            throw new RuntimeException('Could not locate parameter class.');
        }

        $this->cache[$param->id()] = $class::zeros(...$param->param()->shape());
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @internal
     *
     * @param Parameter $param
     * @param \Tensor\Tensor<int|float|array> $gradient
     * @return \Tensor\Tensor<int|float|array>
     */
    public function step(Parameter $param, Tensor $gradient) : Tensor
    {
        $norm = $this->cache[$param->id()];

        $norm = $norm->add($gradient->square());

        $this->cache[$param->id()] = $norm;

        return $gradient->multiply($this->rate)
            ->divide($norm->sqrt()->clipLower(EPSILON));
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
        return "AdaGrad (rate: {$this->rate})";
    }
}
