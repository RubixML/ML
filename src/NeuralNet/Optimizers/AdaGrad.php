<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;
use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use InvalidArgumentException;
use SplObjectStorage;

/**
 * AdaGrad
 *
 * Short for Adaptive Gradient, the AdaGrad Optimizer speeds up the learning of
 * parameters that do not change often and slows down the learning of parameters
 * that do enjoy heavy activity.
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
    public function __construct(float $rate = 0.001)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('The learning rate must be set'
                . ' to a positive value.');
        }

        $this->rate = $rate;
        $this->cache = new SplObjectStorage();
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @param  \Rubix\ML\NeuralNet\Parameter  $parameter
     * @param  \MathPHP\LinearAlgebra\Matrix  $gradients
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Parameter $parameter, Matrix $gradients) : Matrix
    {
        if ($this->cache->contains($parameter)) {
            $cache = $this->cache[$parameter];
        } else {
            $m = $parameter->w()->getM();
            $n = $parameter->w()->getN();

            $cache =  MatrixFactory::zero($m, $n);

            $this->cache->attach($parameter, $cache);
        }

        $cache = $cache->add($gradients->hadamardProduct($gradients));

        $step = [[]];

        foreach ($gradients->getMatrix() as $i => $row) {
            foreach ($row as $j => $gradient) {
                $step[$i][$j] = $this->rate * $gradient
                    / (sqrt($cache[$i][$j]) + self::EPSILON);
            }
        }

        $this->cache[$parameter] = $cache;

        return new Matrix($step);
    }
}
