<?php

namespace Rubix\Engine\Math;

class Random
{
    /**
     * Return a random item from the array.
     *
     * @param  array  $items
     * @return mixed
     */
    public static function item(array $items = [])
    {
        return $items[array_rand($items)];
    }

    /**
     * Generate a random float with given precision i.e. number of decimal places.
     *
     * @param  float  $min
     * @param  float  $max
     * @param  int  $precision
     * @return float
     */
    public static function float(float $min, float $max, int $precision = 2) : float
    {
        $scale = pow(10, $precision);

        return mt_rand($min * $scale, $max * $scale) / $scale;
    }

    /**
     * Generate a random integer.
     *
     * @param  int  $min
     * @param  int  $max
     * @return int
     */
    public static function integer(int $min, int $max) : int
    {
        return mt_rand($min, $max);
    }

    /**
     * Generate a random even integer.
     *
     * @param  int  $min
     * @param  int  $max
     * @return int
     */
    public static function even(int $min, int $max) : int
    {
        $number = self::integer($min, $max);

        if ($number % 2 === 0) {
            return $number;
        } else {
            if ($number + 1 <= $max) {
                return ++$number;
            } else {
                return --$number;
            }
        }
    }

    /**
     * Generate a random odd integer.
     *
     * @param  int  $min
     * @param  int  $max
     * @return int
     */
    public static function odd(int $min, int $max) : int
    {
        $number = self::integer($min, $max);

        if ($number % 2 === 1) {
            return $number;
        } else {
            if ($number + 1 <= $max) {
                return ++$number;
            } else {
                return --$number;
            }
        }
    }
}
