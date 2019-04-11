<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use InvalidArgumentException;

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
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The memoized sum of squared gradient matrices for each parameter.
     *
     * @var \Rubix\Tensor\Matrix[]
     */
    protected $cache;

    /**
     * @param float $rate
     * @throws \InvalidArgumentException
     */
    public function __construct(float $rate = 0.01)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException('The learning rate must be'
                . " greater than 0, $rate given.");
        }

        $this->rate = $rate;
    }
    /**
     * Initialize a parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     */
    public function initialize(Parameter $param) : void
    {
        $g2 = Matrix::zeros(...$param->w()->shape());

        $this->cache[$param->id()] = $g2;
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @param \Rubix\ML\NeuralNet\Parameter $param
     * @param \Rubix\Tensor\Matrix $gradient
     */
    public function step(Parameter $param, Matrix $gradient) : void
    {
        $g2 = $this->cache[$param->id()];

        $g2 = $g2->add($gradient->square());

        $this->cache[$param->id()] = $g2;

        $step = $gradient->multiply($this->rate)
            ->divide($g2->sqrt()->clipLower(self::EPSILON));

        $param->update($step);
    }
}
