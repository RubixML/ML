<?php

namespace Rubix\ML;

use InvalidArgumentException;

use function gettype;
use function in_array;

class DataType
{
    public const OTHER = 0;
    public const CONTINUOUS = 1;
    public const CATEGORICAL = 2;
    public const IMAGE = 3;

    public const TYPES = [
        self::OTHER => 'other',
        self::CONTINUOUS => 'continuous',
        self::CATEGORICAL => 'categorical',
        self::IMAGE => 'image',
    ];

    public const ALL = [
        self::CONTINUOUS,
        self::CATEGORICAL,
        self::IMAGE,
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
            case 'double':
                return self::CONTINUOUS;

            case 'integer':
                return self::CONTINUOUS;

            case 'string':
                return self::CATEGORICAL;

            case 'resource':
                if (get_resource_type($data) === 'gd') {
                    return self::IMAGE;
                } else {
                    return self::OTHER;
                }
                
                // no break
            default:
                return self::OTHER;
        }
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
     * Is the data a resource?
     *
     * @param mixed $data
     * @return bool
     */
    public static function isImage($data) : bool
    {
        return is_resource($data) and get_resource_type($data) === 'gd';
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
            and !self::isImage($data);
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
            throw new InvalidArgumentException('Unknown type given.');
        }

        return self::TYPES[$type];
    }
}
