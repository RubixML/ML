<?php

namespace Rubix\ML\Other\Helpers;

use InvalidArgumentException;

/**
 * Params
 *
 * Generate distributions of values to use in conjunction with Grid Search or
 * other forms of model selection and/or cross validation.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Params
{
    const PHI = 100000000;

    /**
     * Generate a random unique integer distribution.
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
            } else {
                $i--;
            }
        }

        return $distribution;
    }

    /**
     * Generate a random distribution of floating point parameters.
     *
     * @param  float  $min
     * @param  float  $max
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return array
     */
    public static function floats(float $min, float $max, int $n = 10) : array
    {
        if (($max - $min) < 0.0) {
            throw new InvalidArgumentException('Maximum cannot be less than'
                . ' minimum.');
        }

        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate less than 1'
                . ' parameter.');
        }

        $min = (int) round($min * self::PHI);
        $max = (int) round($max * self::PHI);

        $distribution = [];

        for ($i = 0; $i < $n; $i++) {
            $distribution[] = rand($min, $max) / self::PHI;
        }

        return $distribution;
    }

    /**
     * Generate a grid of evenly distributed parameters.
     *
     * @param  float  $start
     * @param  float  $end
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return array
     */
    public static function grid(float $start, float $end, int $n = 10) : array
    {
        if (($end - $start) < 0.0) {
            throw new InvalidArgumentException('End cannot be less than'
                . ' start.');
        }

        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate less than 1'
                . ' parameter.');
        }

        $interval = ($end - $start) / ($n - 1);

        $distribution = [];

        for ($i = $start; $i <= $end; $i += $interval) {
            $distribution[] = $i;
        }

        return $distribution;
    }
}
