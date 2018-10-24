<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\Tensor\Matrix;

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
     * Compute the output value.
     *
     * @param  \Rubix\Tensor\Matrix  $z
     * @return \Rubix\Tensor\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        $activations = [];

        foreach ($z->asColumnVectors() as $vector) {
            $cache = $vector->exp();

            $sigma = $cache->sum() ?: self::EPSILON;

            foreach ($cache as $j => $value) {
                $activations[$j][] = $value / $sigma;
            }
        }

        return Matrix::quick($activations);
    }
}
