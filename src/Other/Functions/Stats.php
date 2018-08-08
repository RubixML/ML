<?php

namespace Rubix\ML\Other\Functions;

use MathPHP\Statistics\Average;

/**
 * Stats
 *
 * Optimized stats functions for common operations.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Stats
{
    /**
     * Compute the population mean and variance and return them in a 2-tuple (2 item
     * array). This function is to be used in place of two separate mean / variance
     * computations as it prevents computing the mean twice.
     *
     * @param  array  $values
     * @return array
     */
    public static function meanVar(array $values) : array
    {
        if (empty($values)) {
            return [0.0, 0.0];
        }

        $mean = Average::mean($values);

        $ssd = 0.0;

        foreach ($values as $value) {
            $ssd += ($value - $mean) ** 2;
        }

        $variance = $ssd / count($values);

        return [$mean, $variance];
    }
}
