<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Parameter;
use Rubix\Tensor\Matrix;
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
     * Precomputed 1. - momentum decay.
     *
     * @var float
     */
    protected $beta1;

    /**
     * Precomputed 1. - RMS decay.
     *
     * @var float
     */
    protected $beta2;

    /**
     * The smoothing parameter. i.e. a tiny number that helps provide numerical
     * smoothing and stability.
     *
     * @var float
     */
    protected $epsilon;

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
     * @param  float  $epsilon
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $rate = 0.001, float $momentumDecay = 0.9,
                                float $rmsDecay = 0.999, float $epsilon = 1e-8)
    {
        if ($rate <= 0.) {
            throw new InvalidArgumentException('The learning rate must be'
                . ' greater than 0.');
        }

        if ($momentumDecay < 0. or $momentumDecay > 1.) {
            throw new InvalidArgumentException('Momentum decay must be between'
                . ' 0 and 1.');
        }

        if ($rmsDecay < 0. or $rmsDecay > 1.) {
            throw new InvalidArgumentException('RMS decay rate must be between'
                . ' 0 and 1.');
        }

        if ($epsilon <= 0.) {
            throw new InvalidArgumentException('Epsilon must be greater than'
                . ' 0.');
        }

        $this->rate = $rate;
        $this->momentumDecay = $momentumDecay;
        $this->rmsDecay = $rmsDecay;
        $this->beta1 = 1. - $momentumDecay;
        $this->beta2 = 1. - $rmsDecay;
        $this->epsilon = $epsilon;
        $this->velocities = new SplObjectStorage();
        $this->cache = new SplObjectStorage();
        $this->t = 0;
    }

    /**
     * Calculate a gradient descent step for a given parameter.
     *
     * @param  \Rubix\ML\NeuralNet\Parameter  $parameter
     * @param  \Rubix\Tensor\Matrix  $gradients
     * @return \Rubix\Tensor\Matrix
     */
    public function step(Parameter $parameter, Matrix $gradients) : Matrix
    {
        if ($this->velocities->contains($parameter)) {
            $velocities = $this->velocities[$parameter];
            $cache = $this->cache[$parameter];
        } else {
            $velocities = Matrix::zeros(...$parameter->w()->shape());
            $cache = Matrix::zeros(...$parameter->w()->shape());

            $this->velocities->attach($parameter, $velocities);
            $this->cache->attach($parameter, $cache);
        }

        $this->t++;

        $velocities = $velocities
            ->multiplyScalar($this->momentumDecay)
            ->add($gradients->multiplyScalar($this->beta1));

        $cache = $cache
            ->multiplyScalar($this->rmsDecay)
            ->add($gradients->square()->multiplyScalar($this->beta2));

        $vHat = $velocities
            ->divideScalar(1. - $this->momentumDecay ** $this->t);

        $rHat = $cache
            ->divideScalar(1. - $this->rmsDecay ** $this->t);

        $step = $vHat
            ->multiplyScalar($this->rate)
            ->divide($rHat->sqrt()->addScalar($this->epsilon));

        $this->velocities[$parameter] = $velocities;
        $this->cache[$parameter] = $cache;

        return $step;
    }
}
