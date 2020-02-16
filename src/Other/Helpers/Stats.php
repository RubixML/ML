<?php

namespace Rubix\ML\Other\Helpers;

use InvalidArgumentException;

use function Rubix\ML\argmax;
use function array_slice;
use function count;

use const Rubix\ML\EPSILON;

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
    /**
     * Compute the population mean of a set of values.
     *
     * @param mixed[] $values
     * @param int|null $n
     * @return float
     */
    public static function mean(array $values, ?int $n = null) : float
    {
        $n = $n ?? count($values);

        if ($n < 1) {
            throw new InvalidArgumentException('Mean is undefined for empty set.');
        }

        return array_sum($values) / $n;
    }

    /**
     * Compute the weighted mean of a set of values.
     *
     * @param mixed[] $values
     * @param mixed[] $weights
     * @return float
     */
    public static function weightedMean(array $values, array $weights) : float
    {
        $n = count($values);

        if ($n < 1) {
            throw new InvalidArgumentException('Weighted mean is undefined for empty set.');
        }

        if (count($weights) !== $n) {
            throw new InvalidArgumentException('The number of weights must'
                . ' equal the number of values.');
        }

        $total = array_sum($weights) ?: EPSILON;

        $sigma = 0.0;

        foreach ($values as $i => $value) {
            $sigma += $value * $weights[$i];
        }

        return $sigma / $total;
    }

    /**
     * Return the midrange of a set of values.
     *
     * @param mixed[] $values
     * @return float
     */
    public static function midrange(array $values) : float
    {
        return (min($values) + max($values)) / 2.0;
    }

    /**
     * Find a mode of a set of values i.e a value that appears most often in the
     * set.
     *
     * @param mixed[] $values
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function mode(array $values) : float
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Mode is undefined for empty set.');
        }

        $counts = array_count_values(array_map('strval', $values));

        return (float) argmax($counts);
    }

    /**
     * Compute the variance of a set of values given a mean and n degrees of
     * freedom.
     *
     * @param mixed[] $values
     * @param float|null $mean
     * @param int|null $n
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function variance(array $values, ?float $mean = null, ?int $n = null) : float
    {
        $n = $n ?? count($values);

        if ($n < 1) {
            throw new InvalidArgumentException('Variance is undefined for an empty set.');
        }

        $mean = $mean ?? self::mean($values);

        $ssd = 0.0;

        foreach ($values as $value) {
            $ssd += ($value - $mean) ** 2;
        }

        return $ssd / $n;
    }

    /**
     * Calculate the median of a set of values.
     *
     * @param mixed[] $values
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function median(array $values) : float
    {
        $n = count($values);

        if ($n < 1) {
            throw new InvalidArgumentException('Median is undefined for empty set.');
        }

        $mid = intdiv($n, 2);

        sort($values);

        if ($n % 2 === 1) {
            $median = $values[$mid];
        } else {
            $median = ($values[$mid - 1] + $values[$mid]) / 2.0;
        }

        return $median;
    }

    /**
     * Calculate the q'th quantile of a given set of values.
     *
     * @param mixed[] $values
     * @param float $q
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function quantile(array $values, float $q) : float
    {
        return (float) current(self::quantiles($values, [$q])) ?: NAN;
    }

    /**
     * Calculate the q'th quantiles of a given set of values.
     *
     * @param mixed[] $values
     * @param float[] $qs
     * @throws \InvalidArgumentException
     * @return (int|float)[]
     */
    public static function quantiles(array $values, array $qs) : array
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Quantile is undefined for empty set.');
        }

        $n = count($values);

        sort($values);

        $quantiles = [];

        foreach ($qs as $q) {
            if ($q < 0.0 or $q > 1.0) {
                throw new InvalidArgumentException('Quantile must be'
                    . " between 0 and 1, $q given.");
            }

            $x = $q * ($n - 1) + 1;

            $xHat = (int) $x;

            $remainder = $x - $xHat;

            $a = $values[$xHat - 1];
            $b = $values[$xHat] ?? end($values);

            $quantiles[] = $a + $remainder * ($b - $a);
        }

        return $quantiles;
    }

    /**
     * Compute the interquartile range of a set of values.
     *
     * @param mixed[] $values
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function iqr(array $values) : float
    {
        $n = count($values);

        if ($n < 1) {
            throw new InvalidArgumentException('Interquartile range is'
                . ' undefined for an empty set.');
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
     * @param mixed[] $values
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
     * @param mixed[] $values
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
            throw new InvalidArgumentException('Central moment is undefined for empty set.');
        }

        if ($moment < 1) {
            throw new InvalidArgumentException('Moment cannot be less than 1.');
        }

        $mean = $mean ?? self::mean($values, $n);

        $sigma = 0.0;

        foreach ($values as $value) {
            $sigma += ($value - $mean) ** $moment;
        }

        return $sigma / $n;
    }

    /**
     * Compute the skewness of a set of values given a mean and n degrees of
     * freedom.
     *
     * @param mixed[] $values
     * @param float|null $mean
     * @param int|null $n
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function skewness(array $values, ?float $mean = null, ?int $n = null) : float
    {
        $n = $n ?? count($values);

        if ($n < 1) {
            throw new InvalidArgumentException('Skewness is undefined for empty set.');
        }

        $mean = $mean ?? self::mean($values, $n);

        $numerator = self::centralMoment($values, 3, $mean);
        $denominator = self::centralMoment($values, 2, $mean) ** 1.5;

        return $numerator / ($denominator ?: EPSILON);
    }

    /**
     * Compute the kurtosis of a set of values.
     *
     * @param mixed[] $values
     * @param float|null $mean
     * @param int|null $n
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function kurtosis(array $values, ?float $mean = null, ?int $n = null) : float
    {
        $n = $n ?? count($values);

        if ($n < 1) {
            throw new InvalidArgumentException('Central moment is undefined for empty set.');
        }

        $mean = $mean ?? self::mean($values, $n);

        $numerator = self::centralMoment($values, 4, $mean, $n);
        $denominator = self::centralMoment($values, 2, $mean, $n) ** 2;

        return $numerator / ($denominator ?: EPSILON) - 3.0;
    }

    /**
     * Return the statistical range given by the maximum minus the minimum
     * of a set of values.
     *
     * @param mixed[] $values
     * @throws \InvalidArgumentException
     * @return float
     */
    public static function range(array $values) : float
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Range is undefined for empty set.');
        }

        return (float) (max($values) - min($values));
    }

    /**
     * Compute the population mean and variance and return them in a 2-tuple.
     *
     * @param mixed[] $values
     * @return float[]
     */
    public static function meanVar(array $values) : array
    {
        $mean = self::mean($values);

        return [$mean, self::variance($values, $mean)];
    }

    /**
     * Compute the population median and median absolute deviation and return
     * them in a 2-tuple.
     *
     * @param mixed[] $values
     * @return float[]
     */
    public static function medianMad(array $values) : array
    {
        $median = self::median($values);

        return [$median, self::mad($values, $median)];
    }
}
