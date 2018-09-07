<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Rubix\ML\Other\Structures\Matrix;
use InvalidArgumentException;

/**
 * Cross Entropy
 *
 * Cross Entropy, or log loss, measures the performance of a classification model
 * whose output is a probability value between 0 and 1. Cross-entropy loss
 * increases as the predicted probability diverges from the actual label. So
 * predicting a probability of .012 when the actual observation label is 1 would
 * be bad and result in a high loss value. A perfect score would have a log loss
 * of 0.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CrossEntropy implements CostFunction
{
    /**
     * The derivative smoothing parameter i.e a small value to add to the
     * denominator of the derivative calculation for numerical stability.
     *
     * @var float
     */
    protected $epsilon;

    /**
     * @param  float  $epsilon
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $epsilon = 1e-10)
    {
        if ($epsilon <= 0.) {
            throw new InvalidArgumentException('Epsilon must be greater than'
                . ' 0.');
        }

        $this->epsilon = $epsilon;
    }

    /**
     * Return a tuple of the min and max output value for this function.
     *
     * @return array
     */
    public function range() : array
    {
        return [0., INF];
    }

    /**
     * Compute the cost.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $expected
     * @param  \Rubix\ML\Other\Structures\Matrix  $activations
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function compute(Matrix $expected, Matrix $activations) : Matrix
    {
        $ones = Matrix::ones(...$activations->shape());

        $eHat = $ones->subtract($expected);
        $aHat = $ones->subtract($activations);

        return $activations->multiply($activations->log())
            ->add($eHat->multiply($aHat->log()))
            ->negate();
    }

    /**
     * Calculate the derivative of the cost function.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $expected
     * @param  \Rubix\ML\Other\Structures\Matrix  $activations
     * @param  \Rubix\ML\Other\Structures\Matrix  $delta
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function differentiate(Matrix $expected, Matrix $activations, Matrix $delta) : Matrix
    {
        $denominator = $activations->subtractScalar(1.)
            ->multiply($activations)
            ->addScalar($this->epsilon);

        return $expected->subtract($activations)
            ->addScalar($this->epsilon)
            ->divide($denominator);
    }
}
