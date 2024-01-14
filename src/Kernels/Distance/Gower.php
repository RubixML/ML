<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * Gower
 *
 * A robust distance kernel that measures samples consisting of a mix of categorical and continuous data
 * types while also handling missing (NaN) values. When comparing continuous data, the Gower metric is
 * equivalent to the normalized [Manhattan](manhattan.md) distance and when comparing categorical data
 * it is equivalent to the [Hamming](hamming.md) distance.
 *
 * > **Note:** The Gower metric expects all continuous variables to have a standardized range. The default
 * range works for values that have been normalized between 0 and 1.
 *
 * References:
 * [1] J. C. Gower. (1971). A General Coefficient of Similarity and Some of Its Properties.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Gower implements Distance, NaNSafe
{
    /**
     * The range of the continuous feature columns.
     *
     * @var float
     */
    protected $range;

    /**
     * @param float $range
     * @throws InvalidArgumentException
     */
    public function __construct(float $range = 1.0)
    {
        if ($range <= 0.0) {
            throw new InvalidArgumentException('Range must be'
                . " greater than 0, $range given.");
        }

        $this->range = $range;
    }

    /**
     * Return the data types that this kernel is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::categorical(),
            DataType::continuous(),
        ];
    }

    /**
     * Compute the distance between two vectors.
     *
     * @param list<string|int|float> $a
     * @param list<string|int|float> $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        $distance = 0.0;
        $numNaNs = 0;

        foreach ($a as $i => $valueA) {
            $valueB = $b[$i];

            switch (true) {
                case is_float($valueA) and is_nan($valueA):
                    ++$numNaNs;

                    break;

                case is_float($valueB) and is_nan($valueB):
                    ++$numNaNs;

                    break;

                case !is_string($valueA) and !is_string($valueB):
                    $distance += abs($valueA - $valueB)
                        / $this->range;

                    break;

                default:
                    if ($valueA !== $valueB) {
                        $distance += 1.0;
                    }
            }
        }

        $n = count($a);

        if ($numNaNs === $n) {
            return NAN;
        }

        return $distance / ($n - $numNaNs);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Gower (range: {$this->range})";
    }
}
