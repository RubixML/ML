<?php

namespace Rubix\ML\Other\Helpers;

use Rubix\ML\Estimator;
use InvalidArgumentException;

use function count;
use function in_array;
use function get_class;
use function gettype;

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
     * @throws \InvalidArgumentException
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
     * Return a string representation of the constructor arguments from
     * an associative constructor array.
     *
     * @param mixed[] $params
     * @param string $separator
     * @return string
     */
    public static function stringify(array $params, string $separator = ' ') : string
    {
        $strings = [];

        foreach ($params as $arg => $param) {
            switch (gettype($param)) {
                case 'object':
                    if ($param instanceof Estimator) {
                        $temp = '(' . self::stringify($param->params(), $separator) . ')';
    
                        $param = self::shortName(get_class($param)) . $temp;
                    } else {
                        $param = self::shortName(get_class($param));
                    }

                    break 1;

                case 'array':
                    $param = '[' . self::stringify($param, $separator) . ']';

                    break 1;

                case 'string':
                    if (class_exists($param)) {
                        $param = self::shortName($param);
                    }

                    break 1;

                case 'boolean':
                    $param = $param ? 'true' : 'false';

                    break 1;

                case 'NULL':
                    $param = 'null';

                    break 1;
            }

            $strings[] = (string) $arg . '=' . (string) $param;
        }

        return implode($separator, $strings);
    }

    /**
     * Return the short class name from a fully qualified class name.
     *
     * @param class-string $class
     * @return string
     */
    public static function shortName(string $class) : string
    {
        return substr(strrchr($class, '\\') ?: '', 1);
    }
}
