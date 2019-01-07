<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
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
    public function __construct(float $rate = 0.001, float $decay = 0.9)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException('The learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($decay < 0. or $decay > 1.) {
            throw new InvalidArgumentException('Decay rate must be between 0'
                . ' and 1.');
        }

        $this->rate = $rate;
        $this->decay = $decay;
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

        $this->cache[$param] = $cache = $cache
            ->multiply($this->decay)
            ->add($gradient->square()->multiply(1. - $this->decay));

        return $gradient->multiply($this->rate)
            ->divide($cache->sqrt()->clipLower(self::EPSILON));
    }
}
