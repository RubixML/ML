<?php

namespace Rubix\ML\Datasets;

class DataType
{
    const OTHER = 0;
    const CONTINUOUS = 1;
    const CATEGORICAL = 2;
    const RESOURCE = 3;

    const TYPES = [
        0 => 'other',
        1 => 'continuous',
        2 => 'categorical',
        3 => 'resource',
    ];

    /**
     * Return the integer encoded data type.
     * 
     * @param  mixed  $data
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
}