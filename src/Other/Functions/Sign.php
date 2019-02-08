<?php

namespace Rubix\ML\Other\Functions;

/**
 * Sign
 *
 * Compute the sign indication of a number.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Sign
{
    /**
     * Compute the sign indication of a number.
     *
     * @param  int|float  $number
     * @return int
     */
    public static function compute($number) : int
    {
        if ($number < 0) {
            return -1;
        } elseif ($number > 0) {
            return 1;
        } else {
            return 0;
        }
    }
}
