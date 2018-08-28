<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Structures\Matrix;
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
     * The smoothing parameter. i.e. a tiny number that helps provide numerical
     * smoothing and stability.
     *
     * @var float
     */
    protected $epsilon;

    /**
     * The memoized sum of squared gradient matrices for each parameter.
     *
     * @var \SplObjectStorage
     */
    protected $cache;

    /**
     * @param  float  $rate
     * @param  float  $epsilon
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, float $epsilon = 1e-8)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException('The learning rate must be'
                . ' greater than 0.');
        }

        if ($epsilon <= 0.) {
            throw new InvalidArgumentException('Epsilon must be greater than'
                . ' 0');
        }

        $this->rate = $rate;
        $this->epsilon = $epsilon;
        $this->cache = new SplObjectStorage();
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @param  \Rubix\ML\NeuralNet\Parameter  $parameter
     * @param  \Rubix\ML\Other\Structures\Matrix  $gradients
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function step(Parameter $parameter, Matrix $gradients) : Matrix
    {
        if ($this->cache->contains($parameter)) {
            $cache = $this->cache[$parameter];
        } else {
            $m = $parameter->w()->m();
            $n = $parameter->w()->n();

            $cache = Matrix::zeros($m, $n);

            $this->cache->attach($parameter, $cache);
        }

        $cache = $cache->add($gradients->square());

        $step = $gradients->scalarMultiply($this->rate)
            ->divide($cache->sqrt()->scalarAdd($this->epsilon));

        $this->cache[$parameter] = $cache;

        return $step;
    }
}
