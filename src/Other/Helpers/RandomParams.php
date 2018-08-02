<?php

namespace Rubix\ML\Other\Helpers;

use MathPHP\Probability\Distribution\Continuous\Uniform;
use InvalidArgumentException;

/**
 * Random Params
 *
 * Generate a unique distribution of values to use in conjunction with
 * Grid Search to randomize the parameter grid.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RandomParams
{
    /**
     * Generate a random parameter distribution of integers.
     *
     * @param  int  $min
     * @param  int  $max
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return array
     */
    public static function ints(int $min, int $max, int $n = 10) : array
    {
        if (($max - $min) < 0) {
            throw new InvalidArgumentException('Maximum cannot be less than'
                . ' minimum.');
        }

        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate less than 1'
                . ' parameter.');
        }

        if ($n > ($max - $min + 1)) {
            throw new InvalidArgumentException('Cannot generate more unique'
                . ' parameters than in range of.');
        }

        $distribution = [];

        for ($i = 0; $i < $n; $i++) {
            $r = rand($min, $max);

            if (!in_array($r, $distribution)) {
                $distribution[] = $r;

                continue 1;
            }

            $i--;
        }

        return $distribution;
    }

    /**
     * Generate a random parameter distribution of floating point numbers.
     *
     * @param  float  $min
     * @param  float  $max
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return array
     */
    public static function floats(float $min, float $max, int $n = 10) : array
    {
        if (($max - $min) < 0) {
            throw new InvalidArgumentException('Maximum cannot be less than'
                . ' minimum.');
        }

        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate less than 1'
                . ' parameter.');
        }

        $distribution = [];

        for ($i = 0; $i < $n; $i++) {
            $distribution[] = rand((int) ($min * 1e8),
                (int) ($max * 1e8)) / 1e8;
        }

        return $distribution;
    }
}
