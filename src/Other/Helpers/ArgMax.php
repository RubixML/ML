<?php

namespace Rubix\ML\Other\Helpers;

/**
 * ArgMax
 *
 * Return the index corresponding to the max value in an array.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ArgMax
{
    /**
     * Compute the argmax of the given values.
     *
     * @param  array  $values
     * @return mixed
     */
    public static function compute(array $values)
    {
        return array_search(max($values), $values);
    }
}
