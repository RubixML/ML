<?php

namespace Rubix\ML\Other\Functions;

/**
 * ArgMax
 *
 * Return the index corresponding to the min value in an array.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ArgMin
{
    /**
     * Compute the argmax of the given values.
     *
     * @param  array  $values
     * @return mixed
     */
    public static function compute(array $values)
    {
        return array_search(min($values), $values);
    }
}
