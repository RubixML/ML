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

    public const TYPE_STRINGS = [
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
     * The integer-encoded data type.
     *
     * @var int
     */
    protected $type;

    /**
     * Build a new data type.
     * @param int $type
     */
    public static function build(int $type) : self
    {
        return new self($type);
    }

    /**
     * Build a data type object from an example value.
     *
     * @param mixed $value
     * @return self
     */
    public static function determine($value) : self
    {
        switch (gettype($value)) {
            case 'double':
            case 'integer':
                return new self(self::CONTINUOUS);

            case 'string':
                return new self(self::CATEGORICAL);

            case 'resource':
                switch (get_resource_type($value)) {
                    case 'gd':
                        return new self(self::IMAGE);

                    default:
                        return new self(self::OTHER);
                }

                // no break
            default:
                return new self(self::OTHER);
        }
    }

    /**
     * Build a continuous data type.
     *
     * @return self
     */
    public static function continuous() : self
    {
        return new self(self::CONTINUOUS);
    }

    /**
     * Build a categorical data type.
     *
     * @return self
     */
    public static function categorical() : self
    {
        return new self(self::CATEGORICAL);
    }

    /**
     * Build an image data type.
     *
     * @return self
     */
    public static function image() : self
    {
        return new self(self::IMAGE);
    }

    /**
     * Build an other data type.
     *
     * @return self
     */
    public static function other() : self
    {
        return new self(self::OTHER);
    }

    /**
     * Return an array with all the data types.
     *
     * @return self[]
     */
    public static function all() : array
    {
        return array_map([self::class, 'build'], self::ALL);
    }

    /**
     * @param int $type
     * @throws \InvalidArgumentException
     */
    public function __construct(int $type)
    {
        if (!in_array($type, self::ALL)) {
            throw new InvalidArgumentException('Invalid type specification.');
        }

        $this->type = $type;
    }

    /**
     * Return the integer-encoded data type.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->type;
    }

    /**
     * Is the data type continuous?
     *
     * @return bool
     */
    public function isContinuous() : bool
    {
        return $this->type === self::CONTINUOUS;
    }

    /**
     * Is the data type categorical?
     *
     * @return bool
     */
    public function isCategorical() : bool
    {
        return $this->type === self::CATEGORICAL;
    }

    /**
     * Is the data type an image resource?
     *
     * @return bool
     */
    public function isImage() : bool
    {
        return $this->type === self::IMAGE;
    }

    /**
     * Does the data not belong to any type?
     *
     * @return bool
     */
    public function isOther() : bool
    {
        return $this->type === self::OTHER;
    }

    /**
     * Return the data type as a string.
     *
     * @return string
     */
    public function __toString() : string
    {
        return self::TYPE_STRINGS[$this->type];
    }
}
