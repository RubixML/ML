<?php

namespace Rubix\ML\Other\Helpers;

use InvalidArgumentException;
use ReflectionMethod;
use ReflectionClass;

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
        if (($max - $min) < 0.) {
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
        if (($end - $start) < 0.) {
            throw new InvalidArgumentException('End cannot be less than'
                . ' start.');
        }

        if ($n < 2) {
            throw new InvalidArgumentException('Cannot generate less than 2'
                . ' parameters.');
        }

        $interval = ($end - $start) / ($n - 1);

        return range($start, $end, $interval);
    }

    /**
     * Extract the arguments from the model constructor for display.
     * 
     * @param  mixed  $object
     * @throws \InvalidArgumentException
     * @return array
     */
    public static function args($object) : array
    {
        if (!is_object($object) and !is_string($object)) {
            throw new InvalidArgumentException('Must provide an object'
                . ' or class name, ' . gettype($object) . ' given.');
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
     * Return the short class name from a fully qualified class name
     * (fqcn).
     * 
     * @param  mixed  $object
     * @throws \InvalidArgumentException
     * @return string
     */
    public static function shortName($object) : string
    {
        if (!is_object($object)) {
            throw new InvalidArgumentException('Must provide an object'
                . gettype($object) . ' given.');
        }

        return substr(strrchr(get_class($object), '\\') ?: '', 1);
    }
}
