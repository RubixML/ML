<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use InvalidArgumentException;

/**
 * RMS Prop
 *
 * An adaptive gradient technique that divides the current gradient over a
 * rolling window of magnitudes of recent gradients.
 *
 * References:
 * [1] T. Tieleman et al. (2012). Lecture 6e rmsprop: Divide the
 * gradient by a running average of its recent magnitude.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RMSProp implements Optimizer, Adaptive
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
     * @var \Rubix\Tensor\Matrix[]
     */
    protected $cache = [
        //
    ];

    /**
     * @param float $rate
     * @param float $decay
     * @throws \InvalidArgumentException
     */
    public function __construct(float $rate = 0.001, float $decay = 0.9)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException('The learning rate must be'
                . " greater than 0, $rate given.");
        }

        if ($decay < 0. or $decay > 1.) {
            throw new InvalidArgumentException('Decay rate must be'
                . " between 0 and 1, $decay given.");
        }

        $this->rate = $rate;
        $this->decay = $decay;
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

        $g2 = $g2->multiply($this->decay)
            ->add($gradient->square()->multiply(1. - $this->decay));

        $this->cache[$param->id()] = $g2;

        $step = $gradient->multiply($this->rate)
            ->divide($g2->sqrt()->clipLower(self::EPSILON));

        $param->update($step);
    }
}
