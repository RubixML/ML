<?php

namespace Rubix\ML\Kernels\Distance;

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
 * > **Note:** This distance kernel is optimized for sparse (mainly zeros) coordinate vectors.
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
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
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
     * @internal
     *
     * @param list<int|float> $a
     * @param list<int|float> $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $sigma = $ssA = $ssB = 0.0;

        foreach ($a as $i => $valueA) {
            $valueB = $b[$i];

            if ($valueA != 0 and $valueB != 0) {
                $sigma += $valueA * $valueB;

                $ssA += $valueA ** 2;
                $ssB += $valueB ** 2;
            } else {
                if ($valueA != 0) {
                    $ssA += $valueA ** 2;
                }

                if ($valueB != 0) {
                    $ssB += $valueB ** 2;
                }
            }
        }

        if ($ssA === 0.0 and $ssB === 0.0) {
            return 0.0;
        }

        if ($ssA === 0.0 or $ssB === 0.0) {
            return 2.0;
        }

        return 1.0 - ($sigma / (sqrt($ssA * $ssB) ?: EPSILON));
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
