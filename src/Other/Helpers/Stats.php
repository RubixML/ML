<?php

namespace Rubix\ML\Other\Helpers;

use Rubix\ML\Other\Functions\Argmax;
use InvalidArgumentException;

/**
 * Stats
 *
 * A helper class providing common statistical functions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Stats
{
    const EPSILON = 1e-10;

    /**
     * Compute the population mean of a set of values.
     *
     * @param  array  $values
     * @param  int|null  $n
     * @return float
     */
    public static function mean(array $values, ?int $n = null) : float
    {
        $n = is_null($n) ? count($values) : $n;

        if ($n === 0) {
            throw new InvalidArgumentException('Mean is undefined for empty'
                . ' set.');
        }

        return array_sum($values) / $n;
    }

    /**
     * Calculate the median of a set of values.
     *
     * @param  array  $values
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function median(array $values) : float
    {
        $n = count($values);

        if ($n === 0) {
            throw new InvalidArgumentException('Median is undefined for empty'
                . ' set.');
        }

        $mid = intdiv($n, 2);

        sort($values);

        if ($n % 2 === 1) {
            $median = $values[$mid];
        } else {
            $median = self::mean([$values[$mid - 1], $values[$mid]]);
        }

        return $median;
    }

    /**
     * Return the midrange of a set of values.
     *
     * @param  array  $values
     * @return float
     */
    public static function midrange(array $values) : float
    {
        return self::mean(self::range($values));
    }

    /**
     * Find a mode of a set of values i.e a value that appears most often in the
     * set.
     *
     * @param  array  $values
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function mode(array $values) : float
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Mode is undefined for empty'
                . ' set.');
        }

        $counts = array_count_values(array_map('strval', $values));

        return (float) Argmax::compute($counts);
    }

    /**
     * Compute the variance of a set of values given a mean and n degrees of
     * freedom.
     *
     * @param  array  $values
     * @param  float|null  $mean
     * @param  int|null  $n
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function variance(array $values, ?float $mean = null, ?int $n = null) : float
    {
        $n = is_null($n) ? count($values) : $n;

        if (empty($values)) {
            throw new InvalidArgumentException('Variance is undefined for an'
                . ' empty set.');
        }

        $mean = is_null($mean) ? self::mean($values) : $mean;

        $ssd = 0.;

        foreach ($values as $value) {
            $ssd += ($value - $mean) ** 2;
        }

        return $ssd / $n;
    }

    /**
     * Compute the interquartile range of a set of values.
     *
     * @param  array  $values
     * @return float
     */
    public static function iqr(array $values) : float
    {
        $n = count($values);

        if ($n === 0) {
            throw new InvalidArgumentException('Interquartile range is not'
                . ' defined for empty set.');
        }

        $mid = intdiv($n, 2);

        sort($values);

        if ($n % 2 === 0) {
            $lower = array_slice($values, 0, $mid);
            $upper = array_slice($values, $mid);
        } else {
            $lower = array_slice($values, 0, $mid);
            $upper = array_slice($values, $mid + 1);
        }

        return self::median($upper) - self::median($lower);
    }

    /**
     * Calculate the median absolute deviation of a set of values given a median.
     *
     * @param  array  $values
     * @param  float|null  $median
     * @return float
     */
    public static function mad(array $values, ?float $median = null) : float
    {
        $median = is_null($median) ? self::median($values) : $median;

        $deviations = [];

        foreach ($values as $value) {
            $deviations[] = abs($value - $median);
        }

        return self::median($deviations);
    }

    /**
     * Compute the n-th central moment of a set of values.
     *
     * @param  array  $values
     * @param  int  $moment
     * @param  float|null  $mean
     * @param  int|null  $n
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function centralMoment(array $values, int $moment, ?float $mean = null, ?int $n = null) : float
    {
        $n = is_null($n) ? count($values) : $n;

        if ($n === 0) {
            throw new InvalidArgumentException('Central moment is undefined for'
                . ' empty set.');
        }

        $mean = is_null($mean) ? self::mean($values, $n) : $mean;

        $sigma = 0.;

        foreach ($values as $value) {
            $sigma += ($value - $mean) ** $moment;
        }

        return $sigma / $n;
    }

    /**
     * Compute the skewness of a set of values given a mean and n degrees of
     * freedom.
     *
     * @param  array  $values
     * @param  float|null  $mean
     * @param  int|null  $n
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function skewness(array $values, ?float $mean = null, ?int $n = null) : float
    {
        $n = is_null($n) ? count($values) : $n;

        if ($n === 0) {
            throw new InvalidArgumentException('Skewness is undefined for'
                . ' empty set.');
        }

        $mean = is_null($mean) ? self::mean($values, $n) : $mean;

        $numerator = self::centralMoment($values, 3, $mean);
        $denominator = self::centralMoment($values, 2, $mean) ** 1.5;

        if ($denominator === 0.) {
            $denominator = self::EPSILON;
        }

        return $numerator / $denominator;
    }

    /**
     * Compute the kurtosis of a set of values.
     *
     * @param  array  $values
     * @param  float|null  $mean
     * @param  int|null  $n
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function kurtosis(array $values, ?float $mean = null, ?int $n = null) : float
    {
        $n = is_null($n) ? count($values) : $n;

        if ($n === 0) {
            throw new InvalidArgumentException('Central moment is undefined for'
                . ' empty set.');
        }

        $mean = is_null($mean) ? self::mean($values, $n) : $mean;

        $numerator = self::centralMoment($values, 4, $mean, $n);
        $denominator = self::centralMoment($values, 2, $mean, $n) ** 2;

        if ($denominator === 0.) {
            $denominator = self::EPSILON;
        }

        return ($numerator / $denominator) - 3.;
    }

    /**
     * Return the minimum and maximum values of a set in a tuple.
     *
     * @param  array  $values
     * @throws \InvalidArgumentException
     * @return array
     */
    public static function range(array $values) : array
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Range is undefined for empty'
                . ' set.');
        }

        return [min($values), max($values)];
    }

    /**
     * Compute the population mean and variance and return them in a 2-tuple.
     *
     * @param  array  $values
     * @return array
     */
    public static function meanVar(array $values) : array
    {
        $mean = self::mean($values);
        $variance = self::variance($values, $mean);

        return [$mean, $variance];
    }

    /**
     * Compute the population median and median absolute deviation and return
     * them in a 2-tuple.
     *
     * @param  array  $values
     * @return array
     */
    public static function medMad(array $values) : array
    {
        $median = self::median($values);
        $mad = self::mad($values, $median);

        return [$median, $mad];
    }
}
