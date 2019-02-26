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
    protected const EPSILON = 1e-8;

    /**
     * Compute the population mean of a set of values.
     *
     * @param array $values
     * @param int|null $n
     * @return float
     */
    public static function mean(array $values, ?int $n = null) : float
    {
        $n = $n ?? count($values);

        if ($n < 1) {
            throw new InvalidArgumentException('Mean is undefined for empty'
                . ' set.');
        }

        return array_sum($values) / $n;
    }

    /**
     * Compute the weighted mean of a set of values.
     *
     * @param array $values
     * @param array $weights
     * @param int|null $n
     * @return float
     */
    public static function weightedMean(array $values, array $weights, ?int $n = null) : float
    {
        $n = $n ?? count($values);

        if ($n < 1) {
            throw new InvalidArgumentException('Mean is undefined for empty'
                . ' set.');
        }

        if (count($weights) !== $n) {
            throw new InvalidArgumentException('The number of weights must'
                . ' equal the number of values.');
        }

        $total = array_sum($weights) ?: self::EPSILON;

        $temp = 0.;

        foreach ($values as $i => $value) {
            $temp += $value * ($weights[$i] ?: self::EPSILON);
        }

        return $temp / $total;
    }

    /**
     * Return the midrange of a set of values.
     *
     * @param array $values
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
     * @param array $values
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
     * @param array $values
     * @param float|null $mean
     * @param int|null $n
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function variance(array $values, ?float $mean = null, ?int $n = null) : float
    {
        $n = $n ?? count($values);

        if ($n < 1) {
            throw new InvalidArgumentException('Variance is undefined for an'
                . ' empty set.');
        }

        $mean = $mean ?? self::mean($values);

        $ssd = 0.;

        foreach ($values as $value) {
            $ssd += ($value - $mean) ** 2;
        }

        return $ssd / $n;
    }

    /**
     * Calculate the median of a set of values.
     *
     * @param array $values
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function median(array $values) : float
    {
        $n = count($values);

        if ($n < 1) {
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
     * Calculate the pth percentile of a given set of values.
     *
     * @param array $values
     * @param float $p
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function percentile(array $values, float $p) : float
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Percentile is not defined for'
                . ' an empty set.');
        }

        if ($p < 0. or $p > 100.) {
            throw new InvalidArgumentException('P must be between 0 and 1'
                . "$p given.");
        }

        $n = count($values);

        sort($values);

        $x = ($p / 100) * ($n - 1) + 1;

        $xHat = (int) $x;

        $remainder = $x - $xHat;

        $a = $values[$xHat - 1];
        $b = $values[$xHat];

        return $a + $remainder * ($b - $a);
    }

    /**
     * Compute the interquartile range of a set of values.
     *
     * @param array $values
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function iqr(array $values) : float
    {
        $n = count($values);

        if ($n < 1) {
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
     * @param array $values
     * @param float|null $median
     * @return float
     */
    public static function mad(array $values, ?float $median = null) : float
    {
        $median = $median ?? self::median($values);

        $deviations = [];

        foreach ($values as $value) {
            $deviations[] = abs($value - $median);
        }

        return self::median($deviations);
    }

    /**
     * Compute the n-th central moment of a set of values.
     *
     * @param array $values
     * @param int $moment
     * @param float|null $mean
     * @param int|null $n
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function centralMoment(array $values, int $moment, ?float $mean = null, ?int $n = null) : float
    {
        $n = $n ?? count($values);

        if ($n < 1) {
            throw new InvalidArgumentException('Central moment is undefined for'
                . ' empty set.');
        }

        $mean = $mean ?? self::mean($values, $n);

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
     * @param array $values
     * @param float|null $mean
     * @param int|null $n
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function skewness(array $values, ?float $mean = null, ?int $n = null) : float
    {
        $n = $n ?? count($values);

        if ($n === 0) {
            throw new InvalidArgumentException('Skewness is undefined for'
                . ' empty set.');
        }

        $mean = $mean ?? self::mean($values, $n);

        $numerator = self::centralMoment($values, 3, $mean);
        $denominator = self::centralMoment($values, 2, $mean) ** 1.5;

        return $numerator / ($denominator ?: self::EPSILON);
    }

    /**
     * Compute the kurtosis of a set of values.
     *
     * @param array $values
     * @param float|null $mean
     * @param int|null $n
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function kurtosis(array $values, ?float $mean = null, ?int $n = null) : float
    {
        $n = $n ?? count($values);

        if ($n === 0) {
            throw new InvalidArgumentException('Central moment is undefined for'
                . ' empty set.');
        }

        $mean = $mean ?? self::mean($values, $n);

        $numerator = self::centralMoment($values, 4, $mean, $n);
        $denominator = self::centralMoment($values, 2, $mean, $n) ** 2;

        return $numerator / ($denominator ?: self::EPSILON) - 3.;
    }

    /**
     * Return the minimum and maximum values of a set in a tuple.
     *
     * @param array $values
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
     * @param array $values
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
     * @param array $values
     * @return array
     */
    public static function medMad(array $values) : array
    {
        $median = self::median($values);
        $mad = self::mad($values, $median);

        return [$median, $mad];
    }
}
