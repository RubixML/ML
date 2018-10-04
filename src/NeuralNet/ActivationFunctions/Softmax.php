<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;
use InvalidArgumentException;

/**
 * Softmax
 *
 * The Softmax function is a generalization of the Sigmoid function that squashes
 * each activation between 0 and 1, and all activations add up to 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Softmax extends Sigmoid
{
    /**
     * The smoothing parameter i.e a small value to add to the denominator for
     * numerical stability.
     *
     * @var float
     */
    protected $epsilon;

    /**
     * @param  float  $epsilon
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $epsilon = 1e-8)
    {
        if ($epsilon <= 0.) {
            throw new InvalidArgumentException('Epsilon must be greater than'
                . ' 0');
        }

        $this->epsilon = $epsilon;
    }

    /**
     * Compute the output value.
     *
     * @param  \Rubix\Tensor\Matrix  $z
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        $activations = [[]];

        foreach ($z->transpose()->asVectors() as $i => $vector) {
            $cache = $vector->exp();

            $sigma = $cache->sum() + $this->epsilon;

            foreach ($cache as $j => $value) {
                $activations[$j][$i] = $value / $sigma;
            }
        }

        return new Matrix($activations, false);
    }
}
