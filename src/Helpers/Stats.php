<?php

namespace Rubix\ML\Helpers;

use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;
use function array_sum;
use function sort;
use function abs;

use const Rubix\ML\EPSILON;

/**
 * Stats
 *
 * A helper class providing statistical functions.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Stats
{
    /**
     * Compute the mean of a set of values.
     *
     * @param mixed[] $values
     * @return float
     */
    public static function mean(array $values) : float
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Mean is undefined for empty set.');
        }

        return array_sum($values) / count($values);
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
        if (empty($values)) {
            throw new InvalidArgumentException('Weighted mean is undefined for empty set.');
        }

        if (count($values) !== count($weights)) {
            throw new InvalidArgumentException('The number of weights must'
                . ' equal the number of values.');
        }

        $total = array_sum($weights);

        if ($total == 0) {
            throw new InvalidArgumentException('Total weight cannot equal 0.');
        }

        $sigma = 0.0;

        foreach ($values as $i => $value) {
            $sigma += $value * $weights[$i];
        }

        return $sigma / $total;
    }

    /**
     * Calculate the median of a set of values.
     *
     * @param mixed[] $values
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float
     */
    public static function median(array $values) : float
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Median is undefined for empty set.');
        }

        $n = count($values);

        $mid = intdiv($n, 2);

        sort($values);

        if ($n % 2 === 1) {
            $median = $values[$mid];
        } else {
            $median = 0.5 * ($values[$mid - 1] + $values[$mid]);
        }

        return $median;
    }

    /**
     * Calculate the q'th quantile of a given set of values.
     *
     * @param mixed[] $values
     * @param float $q
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float
     */
    public static function quantile(array $values, float $q) : float
    {
        return current(self::quantiles($values, [$q])) ?: NAN;
    }

    /**
     * Calculate the q'th quantiles of a given set of values.
     *
     * @param mixed[] $values
     * @param float[] $qs
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
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
     * Compute the variance of a set of values.
     *
     * @param mixed[] $values
     * @param float|null $mean
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float
     */
    public static function variance(array $values, ?float $mean = null) : float
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Variance is undefined for empty set.');
        }

        $mean = $mean ?? self::mean($values);

        $ssd = 0.0;

        foreach ($values as $value) {
            $ssd += ($value - $mean) ** 2;
        }

        return $ssd / count($values);
    }

    /**
     * Calculate the median absolute deviation of a set of values.
     *
     * @param mixed[] $values
     * @param float|null $median
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float
     */
    public static function mad(array $values, ?float $median = null) : float
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Median absolute deviation'
                . ' is undefined for empty set.');
        }

        $median = $median ?? self::median($values);

        $deviations = [];

        foreach ($values as $value) {
            $deviations[] = abs($value - $median);
        }

        return self::median($deviations);
    }

    /**
     * Compute the skewness of a set of values.
     *
     * @param mixed[] $values
     * @param float|null $mean
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float
     */
    public static function skewness(array $values, ?float $mean = null) : float
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Skewness is undefined for empty set.');
        }

        $mean = $mean ?? self::mean($values);

        $numerator = self::centralMoment($values, 3, $mean);
        $denominator = self::centralMoment($values, 2, $mean) ** 1.5;

        return $numerator / ($denominator ?: EPSILON);
    }

    /**
     * Compute the kurtosis of a set of values.
     *
     * @param mixed[] $values
     * @param float|null $mean
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float
     */
    public static function kurtosis(array $values, ?float $mean = null) : float
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Kurtosis is undefined for empty set.');
        }

        $mean = $mean ?? self::mean($values);

        $numerator = self::centralMoment($values, 4, $mean);
        $denominator = self::centralMoment($values, 2, $mean) ** 2;

        return $numerator / ($denominator ?: EPSILON) - 3.0;
    }

    /**
     * Compute the n-th central moment of a set of values.
     *
     * @param mixed[] $values
     * @param int $moment
     * @param float|null $mean
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float
     */
    public static function centralMoment(array $values, int $moment, ?float $mean = null) : float
    {
        if (empty($values)) {
            throw new InvalidArgumentException('Central moment is undefined for empty set.');
        }

        if ($moment < 1) {
            throw new InvalidArgumentException('Moment cannot be less than 1.');
        }

        $mean = $mean ?? self::mean($values);

        $sigma = 0.0;

        foreach ($values as $value) {
            $sigma += ($value - $mean) ** $moment;
        }

        return $sigma / count($values);
    }

    /**
     * Compute the mean and variance and return them in a 2-tuple.
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
     * Compute the median and median absolute deviation and return them in a 2-tuple.
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
