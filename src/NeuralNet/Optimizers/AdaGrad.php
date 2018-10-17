<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use InvalidArgumentException;
use SplObjectStorage;

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
class AdaGrad implements Optimizer
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
     * @var \SplObjectStorage
     */
    protected $cache;

    /**
     * @param  float  $rate
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.01)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException("The learning rate must be"
                . " greater than 0, $rate given.");
        }

        $this->rate = $rate;
        $this->cache = new SplObjectStorage();
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @param  \Rubix\ML\NeuralNet\Parameter  $param
     * @param  \Rubix\Tensor\Matrix  $gradient
     * @return \Rubix\Tensor\Matrix
     */
    public function step(Parameter $param, Matrix $gradient) : Matrix
    {
        if ($this->cache->contains($param)) {
            $cache = $this->cache[$param];
        } else {
            $cache = Matrix::zeros(...$param->w()->shape());

            $this->cache->attach($param, $cache);
        }

        $this->cache[$param] = $cache = $cache->add($gradient->square());

        $step = $gradient->multiply($this->rate)
            ->divide($cache->sqrt()->clip(self::EPSILON, INF));

        return $step;
    }
}
