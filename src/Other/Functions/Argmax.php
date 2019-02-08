<?php

namespace Rubix\ML\Other\Functions;

use function array_search;
use function max;

/**
 * Argmax
 *
 * Return the index corresponding to the max value in an array.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Argmax
{
    /**
     * Compute the argmax of the given values.
     *
     * @param array $values
     * @return mixed
     */
    public static function compute(array $values)
    {
        return $values ? array_search(max($values), $values) : null;
    }
}
