<?php

namespace Rubix\ML\NeuralNet\ActivationFunctions;

use Rubix\ML\Other\Structures\Matrix;

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
     * @param  \Rubix\ML\Other\Structures\Matrix  $z
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function compute(Matrix $z) : Matrix
    {
        $activations = [[]];

        foreach ($z->transpose()->asArray() as $i => $vector) {
            $cache = [];

            foreach ($vector as $j => $value) {
                $cache[$j] = exp($value);
            }

            $sigma = array_sum($cache) + self::EPSILON;

            foreach ($cache as $j => $value) {
                $activations[$j][$i] = $value / $sigma;
            }
        }

        return new Matrix($activations, false);
    }
}
