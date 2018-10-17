<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
use InvalidArgumentException;
use SplObjectStorage;

/**
 * Adam
 *
 * Short for Adaptive Momentum Estimation, the Adam Optimizer uses both Momentum
 * and RMS properties to achieve a balance of speed and stability.
 *
 * References:
 * [1] D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Adam implements Optimizer
{
    /**
     * The learning rate. i.e. the master step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * The decay rate of the momentum property.
     *
     * @var float
     */
    protected $momentumDecay;

    /**
     * The decay rate of the RMS property.
     *
     * @var float
     */
    protected $rmsDecay;

    /**
     * The memoized velocity matrices.
     *
     * @var \SplObjectStorage
     */
    protected $velocities;

    /**
     * The rolling sum of squared gradient matrices.
     *
     * @var \SplObjectStorage
     */
    protected $cache;

    /**
     * The number of steps taken since instantiation.
     *
     * @var int
     */
    protected $t;

    /**
     * @param  float  $rate
     * @param  float  $momentumDecay
     * @param  float  $rmsDecay
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, float $momentumDecay = 0.9, float $rmsDecay = 0.999)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException("The learning rate must be"
                . " greater than 0, $rate given.");
        }

        if ($momentumDecay < 0. or $momentumDecay > 1.) {
            throw new InvalidArgumentException('Momentum decay must be between'
                . ' 0 and 1.');
        }

        if ($rmsDecay < 0. or $rmsDecay > 1.) {
            throw new InvalidArgumentException('RMS decay rate must be between'
                . ' 0 and 1.');
        }

        $this->rate = $rate;
        $this->momentumDecay = $momentumDecay;
        $this->rmsDecay = $rmsDecay;
        $this->velocities = new SplObjectStorage();
        $this->cache = new SplObjectStorage();
        $this->t = 0;
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
        if ($this->velocities->contains($param)) {
            $velocities = $this->velocities[$param];
            $cache = $this->cache[$param];
        } else {
            $velocities = Matrix::zeros(...$param->w()->shape());
            $cache = Matrix::zeros(...$param->w()->shape());

            $this->velocities->attach($param, $velocities);
            $this->cache->attach($param, $cache);
        }

        $this->t++;

        $this->velocities[$param] = $velocities = $velocities
            ->multiply($this->momentumDecay)
            ->add($gradient->multiply(1. - $this->momentumDecay));

        $this->cache[$param] = $cache = $cache
            ->multiply($this->rmsDecay)
            ->add($gradient->square()->multiply(1. - $this->rmsDecay));

        $vHat = $velocities->divide(1. - $this->momentumDecay ** $this->t);

        $rHat = $cache->divide(1. - $this->rmsDecay ** $this->t);

        return $vHat->multiply($this->rate)
            ->divide($rHat->sqrt()->clip(self::EPSILON, INF));

    }
}
