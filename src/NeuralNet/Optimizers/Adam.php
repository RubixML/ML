<?php

namespace Rubix\ML\NeuralNet\Optimizers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use InvalidArgumentException;

/**
 * Adam
 *
 * Short for Adaptive Momentum Estimation, the Adam Optimizer uses both Momentum
 * and RMS properties to achieve a balance of velocity and stability.
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
     * The smoothing parameter. i.e. a tiny number that helps provide numerical
     * smoothing and stability.
     *
     * @var float
     */
    protected $epsilon;

    /**
     * The velocity matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $velocities;

    /**
     * The RMS matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $cache;

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
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('The learning rate must be set'
                . ' to a positive value.');
        }

        if ($momentumDecay < 0.0 or $momentumDecay > 1.0) {
            throw new InvalidArgumentException('Momentum decay must be between'
                . ' 0 and 1.');
        }

        if ($rmsDecay < 0.0 or $rmsDecay > 1.0) {
            throw new InvalidArgumentException('RMS decay rate must be between'
                . ' 0 and 1.');
        }

        if ($epsilon === 0.0) {
            throw new InvalidArgumentException('Epsilon cannot be 0.');
        }

        $this->rate = $rate;
        $this->momentumDecay = $momentumDecay;
        $this->rmsDecay = $rmsDecay;
        $this->epsilon = $epsilon;
    }

    /**
     * Initialize the layer optimizer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $weights
     * @return void
     */
    public function initialize(Matrix $weights) : void
    {
        $this->velocities = MatrixFactory::zero($weights->getM(), $weights->getN());

        $this->cache = MatrixFactory::zero($weights->getM(), $weights->getN());
    }

    /**
     * Calculate a gradient descent step for a layer given a matrix of gradients.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $gradients
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function step(Matrix $gradients) : Matrix
    {
        $this->velocities = $this->velocities->scalarMultiply($this->momentumDecay)
            ->add($gradients->scalarMultiply(1 - $this->momentumDecay));

        $this->cache = $this->cache->scalarMultiply($this->rmsDecay)
            ->add($gradients->hadamardProduct($gradients->scalarMultiply(1 - $this->rmsDecay)));

        $m = $gradients->getM();
        $n = $gradients->getN();

        $steps = [[]];

        for ($i = 0; $i < $m; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $steps[$i][$j] = $this->rate * $this->velocities[$i][$j]
                    / (sqrt($this->cache[$i][$j]) + $this->epsilon);
            }
        }

        return new Matrix($steps);
    }
}
