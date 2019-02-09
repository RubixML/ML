<?php

namespace Rubix\ML\Other\Functions;

/**
 * LogSumExp
 *
 * The LogSumExp (LSE) function is a smooth approximation to the max function.
 * It's defined as the logarithm of the sum of the exponentials of the
 * arguments.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LogSumExp
{
    /**
     * Compute the log of the sum of exponential values.
     *
     * @param array $values
     * @return float|null
     */
    public static function compute(array $values) : ?float
    {
        return empty($values) ? null : log(array_sum(array_map('exp', $values)));
    }
}
