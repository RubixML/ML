<?php

namespace Rubix\ML\Other\Helpers;

use InvalidArgumentException;

use function gettype;
use function is_string;
use function is_int;
use function is_float;
use function is_numeric;
use function is_resource;

class DataType
{
    public const OTHER = 0;
    public const CONTINUOUS = 1;
    public const CATEGORICAL = 2;
    public const RESOURCE = 3;

    public const TYPES = [
        self::OTHER => 'other',
        self::CONTINUOUS => 'continuous',
        self::CATEGORICAL => 'categorical',
        self::RESOURCE => 'resource',
    ];

    public const ALL = [
        self::CONTINUOUS,
        self::CATEGORICAL,
        self::RESOURCE,
        self::OTHER,
    ];

    /**
     * Return the integer encoded data type.
     *
     * @param mixed $data
     * @return int
     */
    public static function determine($data) : int
    {
        switch (gettype($data)) {
            case 'string':
                return self::CATEGORICAL;

            case 'double':
                return self::CONTINUOUS;

            case 'integer':
                return self::CONTINUOUS;

            case 'resource':
                return self::RESOURCE;

            default:
                return self::OTHER;
        }
    }

    /**
     * Is the data categorical?
     *
     * @param mixed $data
     * @return bool
     */
    public static function isCategorical($data) : bool
    {
        return is_string($data);
    }

    /**
     * Is the data continuous?
     *
     * @param mixed $data
     * @return bool
     */
    public static function isContinuous($data) : bool
    {
        return is_int($data) or is_float($data);
    }

    /**
     * Is the data a resource?
     *
     * @param mixed $data
     * @return bool
     */
    public static function isResource($data) : bool
    {
        return is_resource($data);
    }

    /**
     * Does the data not belong to any type?
     *
     * @param mixed $data
     * @return bool
     */
    public static function isOther($data) : bool
    {
        return !is_string($data)
            and !is_numeric($data)
            and !is_resource($data);
    }

    /**
     * Return the integer type as a string.
     *
     * @param int $type
     * @throws \InvalidArgumentException
     * @return string
     */
    public static function asString(int $type) : string
    {
        if (!in_array($type, self::ALL)) {
            throw new InvalidArgumentException('Unkown type given.');
        }

        return self::TYPES[$type];
    }
}
