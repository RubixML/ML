<?php

namespace Rubix\ML\Helpers;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Stringable;

use function count;
use function in_array;
use function get_class;
use function gettype;
use function max;
use function abs;

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
    /**
     * Generate a random unique integer distribution.
     *
     * @param int $min
     * @param int $max
     * @param int $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return int[]
     */
    public static function ints(int $min, int $max, int $n = 10) : array
    {
        if ($max <= $min) {
            throw new InvalidArgumentException('Maximum cannot be'
                . ' less than or equal to minimum.');
        }

        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate less'
                . ' than 1 parameter.');
        }

        if ($n > ($max - $min + 1)) {
            throw new InvalidArgumentException('Cannot generate more'
                . ' unique integers than within range of.');
        }

        $dist = [];

        while (count($dist) < $n) {
            $r = rand($min, $max);

            if (!in_array($r, $dist)) {
                $dist[] = $r;
            }
        }

        return $dist;
    }

    /**
     * Generate a random distribution of floating point parameters.
     *
     * @param float $min
     * @param float $max
     * @param int $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float[]
     */
    public static function floats(float $min, float $max, int $n = 10) : array
    {
        if ($max <= $min) {
            throw new InvalidArgumentException('Maximum cannot be'
                . ' less than or equal to minimum.');
        }

        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate'
                . ' less than 1 parameter.');
        }

        $phi = getrandmax() / max(abs($max), abs($min));

        $min = (int) floor($min * $phi);
        $max = (int) ceil($max * $phi);

        $dist = [];

        while (count($dist) < $n) {
            $dist[] = rand($min, $max) / $phi;
        }

        return $dist;
    }

    /**
     * Generate a grid of evenly distributed parameters.
     *
     * @param float $min
     * @param float $max
     * @param int $n
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return float[]
     */
    public static function grid(float $min, float $max, int $n = 10) : array
    {
        if ($max <= $min) {
            throw new InvalidArgumentException('Maximum cannot be'
                . ' less than or equal to minimum.');
        }

        if ($n < 2) {
            throw new InvalidArgumentException('Cannot generate less'
                . ' than 2 parameters.');
        }

        $interval = ($max - $min) / ($n - 1);

        return range($min, $max, $interval);
    }

    /**
     * Return a string representation of the constructor arguments from an associative
     * constructor array.
     *
     * @internal
     *
     * @param mixed[] $params
     * @param string $separator
     * @return string
     */
    public static function stringify(array $params, string $separator = ', ') : string
    {
        $strings = [];

        foreach ($params as $arg => $param) {
            $strings[] = $arg . ': ' . self::toString($param);
        }

        return implode($separator, $strings);
    }

    /**
     * Convert the value of an argument to its string representation.
     *
     * @internal
     *
     * @param mixed $value
     * @return string
     */
    public static function toString($value) : string
    {
        switch (gettype($value)) {
            case 'object':
                if ($value instanceof Stringable) {
                    return (string) $value;
                }

                return self::shortName(get_class($value));

            case 'array':
                return '[' . self::stringify($value, ', ') . ']';

            case 'string':
                if (class_exists($value)) {
                    return self::shortName($value);
                }

                break;

            case 'integer':
            case 'double':
                return (string) $value;

            case 'boolean':
                return $value ? 'true' : 'false';

            case 'NULL':
                return 'null';
        }

        return $value;
    }

    /**
     * Return the short class name from a fully qualified class name.
     *
     * @internal
     *
     * @param class-string $class
     * @return string
     */
    public static function shortName(string $class) : string
    {
        return substr(strrchr($class, '\\') ?: '', 1);
    }
}
