<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;
use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use InvalidArgumentException;
use SplObjectStorage;

/**
 * RMS Prop
 *
 * An adaptive gradient technique that divides the current gradient over a
 * rolling window of magnitudes of recent gradients.
 *
 * References:
 * [1] T. Tieleman and G. E. Hinton. (2012). Lecture 6e rmsprop: Divide the
 * gradient by a running average of its recent magnitude.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RMSProp implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The rms decay rate.
     *
     * @var float
     */
    protected $decay;

    /**
     * The smoothing parameter. i.e. a tiny number that helps provide numerical
     * smoothing and stability.
     *
     * @var float
     */
    protected $epsilon;

    /**
     * The rolling sum of squared gradient matrices.
     *
     * @var \SplObjectStorage
     */
    protected $cache;

    /**
     * @param  float  $rate
     * @param  float  $decay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, float $decay = 0.9, float $epsilon = 1e-8)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException('The learning rate must be'
                . ' greater than 0.');
        }

        if ($decay < 0. or $decay > 1.) {
            throw new InvalidArgumentException('Decay rate must be between 0'
                . ' and 1.');
        }

        if ($epsilon <= 0.) {
            throw new InvalidArgumentException('Epsilon must be greater than'
                . ' 0');
        }

        $this->rate = $rate;
        $this->decay = $decay;
        $this->epsilon = $epsilon;
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

            $cache = MatrixFactory::zero($m, $n);

            $this->cache->attach($parameter, $cache);
        }

        $cache = $cache
            ->scalarMultiply($this->decay)
            ->add($gradients->hadamardProduct($gradients)->scalarMultiply(1 - $this->decay));

        $steps = [[]];

        foreach ($gradients->getMatrix() as $i => $row) {
            foreach ($row as $j => $gradient) {
                $steps[$i][$j] = $this->rate * $gradient
                    / ($cache[$i][$j] ** 0.5 + $this->epsilon);
            }
        }

        $this->cache[$parameter] = $cache;

        return new Matrix($steps);
    }
}
