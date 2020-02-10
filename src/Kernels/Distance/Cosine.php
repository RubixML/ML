<?php

namespace Rubix\ML\Kernels\Distance;

use Tensor\Vector;
use Rubix\ML\DataType;

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
class Cosine implements Distance
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
        $a = Vector::quick($a);
        $b = Vector::quick($b);

        return 1.0 - ($a->dot($b) / (($a->l2Norm() * $b->l2Norm()) ?: EPSILON));
    }
}
