<?php

namespace Rubix\Engine;

use InvalidArgumentException;
use RuntimeException;

class Stats
{
    /**
     * Return the sum of a given property. O(N)
     *
     * @param  array  $values
     * @return mixed
     */
    public static function sum(array $values)
    {
        return array_sum($values);
    }

    /**
     * Return the mean of a given property or null if empty set.
     *
     * @param  array  $values
     * @return mixed|null
     */
    public static function mean(array $values)
    {
        $count = count($values);

        return $count > 0 ? self::sum($values) / $count : null;
    }

    /**
     * Alias of mean.
     *
     * @param  array  $values
     * @return mixed|null
     */
    public static function average(array $values)
    {
        return self::mean($values);
    }

    /**
     * Return the median of a given property or null for empty set. If a middle
     * value cannot be determined, the user can specify to return the LOW, HIGH,
     * or AVERAGE value of the two.
     *
     * @param  array  $values
     * @param  string  $split
     * @return mixed
     */
    public static function median(array $values, string $split = 'AVERAGE')
    {
        $count = count($values);

        if ($count === 0) {
            return null;
        }

        sort($values);

        $middle = floor($count / 2) - 1;

        if ($count % 2) {
            return $values[$middle];
        } else {
            if ($split === 'LOWER') {
                return $values[$middle];
            } else if ($split === 'UPPER') {
                return $values[$middle + 1];
            } else {
                return ($values[$middle] + $values[$middle + 1]) / 2;
            }
        }
    }

    /**
     * Return the mode of a given property. Valid for numerics or strings.
     *
     * @param  array  $values
     * @return mixed
     */
    public static function mode(array $values)
    {
        $values = array_count_values($values);

        return array_search(max($values), $values);
    }

    /**
     * Compute the variance of a property.
     *
     * @param  array  $values
     * @return mixed
     */
    public static function variance(array $values)
    {
        $mean = self::mean($values);

        return array_reduce($values, function ($carry, $value) use ($mean) {
            return $carry += pow($value - $mean, 2);
        }, 0) / count($values);
    }

    /**
     * Compute the standard deviation of a property.
     *
     * @param  array  $values
     * @return mixed
     */
    public static function stddev(array $values)
    {
        return sqrt(self::variance($values));
    }

    /**
     * Return the minimum value of a given property.
     *
     * @param  array  $values
     * @return mixed
     */
    public static function min(array $values)
    {
        return min($values);
    }

    /**
     * Return the maximum value of a given property.
     *
     * @param  array  $values
     * @return mixed
     */
    public static function max(array $values)
    {
        return max($values);
    }

    /**
     * Return a softmax array.
     *
     * @param  array  $values
     * @return mixed
     */
    public static function softMax(array $values) : array
    {
        $sum = self::sum($values);

        foreach ($values as $i => $value) {
            $values[$i] = $value / $sum;
        }

        return $values;
    }

    /**
     * Format the number to a human readable format.
     *
     * @param  mixed  $number
     * @param  int  $precision
     * @return string
     */
    public static function format($number, int $precision = 0) : string
    {
        return number_format($number, $precision);
    }

    /**
     * Round a number to the nearest decimal place.
     *
     * @param  mixed  $number
     * @param  int  $precision
     * @return mixed
     */
    public static function round($number, int $precision = 0)
    {
        return round($number, $precision);
    }
}
