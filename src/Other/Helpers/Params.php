<?php

namespace Rubix\ML\Other\Helpers;

use InvalidArgumentException;
use ReflectionMethod;
use ReflectionClass;

use const Rubix\ML\PHI;

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
     * @throws \InvalidArgumentException
     * @return int[]
     */
    public static function ints(int $min, int $max, int $n = 10) : array
    {
        if (($max - $min) < 0) {
            throw new InvalidArgumentException('Maximum cannot be'
                . ' less than minimum.');
        }

        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate less'
                . ' than 1 parameter.');
        }

        if ($n > ($max - $min + 1)) {
            throw new InvalidArgumentException('Cannot generate more'
                . ' unique integers than within range.');
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
     * @throws \InvalidArgumentException
     * @return float[]
     */
    public static function floats(float $min, float $max, int $n = 10) : array
    {
        if (($max - $min) < 0.) {
            throw new InvalidArgumentException('Maximum cannot be'
                . ' less than minimum.');
        }

        if ($n < 1) {
            throw new InvalidArgumentException('Cannot generate less'
                . ' than 1 parameter.');
        }

        $min = (int) round($min * PHI);
        $max = (int) round($max * PHI);

        $dist = [];

        while (count($dist) < $n) {
            $dist[] = rand($min, $max) / PHI;
        }

        return $dist;
    }

    /**
     * Generate a grid of evenly distributed parameters.
     *
     * @param float $min
     * @param float $max
     * @param int $n
     * @throws \InvalidArgumentException
     * @return float[]
     */
    public static function grid(float $min, float $max, int $n = 10) : array
    {
        if ($min > $max) {
            throw new InvalidArgumentException('Max cannot be less'
                . ' then min.');
        }

        if ($n < 2) {
            throw new InvalidArgumentException('Cannot generate less'
                . ' than 2 parameters.');
        }

        $interval = ($max - $min) / ($n - 1);

        return range($min, $max, $interval);
    }

    /**
     * Extract the arguments from the model constructor for display.
     *
     * @param object $object
     * @throws \InvalidArgumentException
     * @return string[]
     */
    public static function args($object) : array
    {
        if (!is_object($object)) {
            throw new InvalidArgumentException('Argument must be'
                . ' an object ' . gettype($object) . ' found.');
        }

        $reflector = new ReflectionClass($object);

        $constructor = $reflector->getConstructor();

        if ($constructor instanceof ReflectionMethod) {
            $args = array_column($constructor->getParameters(), 'name');
        } else {
            $args = [];
        }

        return $args;
    }

    /**
     * Return a string representation of the constructor arguments from
     * an associative constructor array.
     *
     * @param array $constructor
     * @param string $equator
     * @param string $separator
     * @return string
     */
    public static function stringify(array $constructor, string $equator = '=', string $separator = ' ') : string
    {
        $strings = [];

        foreach ($constructor as $arg => $param) {
            if (is_object($param)) {
                $param = self::shortName($param);
            }

            if (is_array($param)) {
                $temp = array_combine(array_keys($param), $param) ?: [];

                $param = '[' . self::stringify($temp) . ']';
            }

            $strings[] = (string) $arg . $equator . (string) $param;
        }

        return implode($separator, $strings);
    }

    /**
     * Return the short class name from a fully qualified class name (fqcn).
     *
     * @param mixed $object
     * @throws \InvalidArgumentException
     * @return string
     */
    public static function shortName($object) : string
    {
        if (!is_object($object) and !is_string($object)) {
            throw new InvalidArgumentException('Must provide an object'
                . ' or class string ' . gettype($object) . ' given.');
        }

        $class = is_object($object) ? get_class($object) : $object;

        return substr(strrchr($class, '\\') ?: '', 1);
    }
}
