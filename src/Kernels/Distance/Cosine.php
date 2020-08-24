<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\DataType;
use Stringable;

use const Rubix\ML\EPSILON;

/**
 * Cosine
 *
 * Cosine Similarity is a measure that ignores the magnitude of the distance
 * between two vectors thus acting as strictly a judgement of orientation. Two
 * vectors with the same orientation have a cosine similarity of 1, two vectors
 * oriented at 90° relative to each other have a similarity of 0, and two
 * vectors diametrically opposed have a similarity of -1. To be used as a
 * distance kernel, we subtract the Cosine Similarity from 1 in order to
 * satisfy the positive semi-definite condition, therefore the Cosine distance
 * is a number between 0 and 2.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Cosine implements Distance, Stringable
{
    /**
     * Return the data types that this kernel is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Compute the distance between two vectors.
     *
     * @param (int|float)[] $a
     * @param (int|float)[] $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $sigma = $ssA = $ssB = 0.0;

        foreach ($a as $i => $valueA) {
            $valueB = $b[$i];

            $sigma += $valueA * $valueB;

            $ssA += $valueA ** 2;
            $ssB += $valueB ** 2;
        }

        return 1.0 - ($sigma / ((sqrt($ssA) * sqrt($ssB)) ?: EPSILON));
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Cosine';
    }
}
