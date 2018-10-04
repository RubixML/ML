<?php

namespace Rubix\ML\Other\Helpers;

/**
 * Gaussian
 *
 * Functions involving standard normal (Gaussian) distributions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Gaussian
{
    const TWO_PI = 2. * M_PI;

    /**
     * Generate a random number from a Gaussian distribution with 0 mean and
     * standard deviation of 1 i.e a number between -1 and 1.
     *
     * @return float
     */
    public static function rand() : float
    {
        $r1 = rand(0, PHP_INT_MAX) / PHP_INT_MAX;
        $r2 = rand(0, PHP_INT_MAX) / PHP_INT_MAX;

        return sqrt(-2. * log($r1)) * cos(self::TWO_PI * $r2);
    }
}
