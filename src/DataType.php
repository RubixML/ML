<?php

namespace Rubix\ML;

use Rubix\ML\Exceptions\InvalidArgumentException;
use Stringable;
use GdImage;

use function gettype;
use function in_array;

/**
 * Data Type
 *
 * A high-level data type value object.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DataType implements Stringable
{
    /**
     * The continuous data type code.
     *
     * @var int
     */
    public const CONTINUOUS = 1;

    /**
     * The categorical data type code.
     *
     * @var int
     */
    public const CATEGORICAL = 2;

    /**
     * The image data type code.
     *
     * @var int
     */
    public const IMAGE = 3;

    /**
     * Any other data type that is not supported natively.
     *
     * @var int
     */
    public const OTHER = 0;

    /**
     * The human-readable string representations of the high-level data types.
     *
     * @var literal-string[]
     */
    protected const TYPE_STRINGS = [
        self::OTHER => 'other',
        self::CONTINUOUS => 'continuous',
        self::CATEGORICAL => 'categorical',
        self::IMAGE => 'image',
    ];

    /**
     * An array of all the high-level data type codes.
     *
     * @var list<int>
     */
    protected const ALL = [
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
    protected int $code;

    /**
     * Build a new data type object.
     *
     * @param int $code
     */
    public static function build(int $code) : self
    {
        return new self($code);
    }

    /**
     * Build a data type object from an example value.
     *
     * @param mixed $value
     * @return self
     */
    public static function detect($value) : self
    {
        switch (gettype($value)) {
            case 'double':
            case 'integer':
                return new self(self::CONTINUOUS);

            case 'string':
                return new self(self::CATEGORICAL);

            case 'object':
                if (class_exists(GdImage::class) and $value instanceof GdImage) {
                    return new self(self::IMAGE);
                }

                return new self(self::OTHER);

            case 'resource':
                switch (get_resource_type($value)) {
                    case 'gd':
                        return new self(self::IMAGE);
                }

                return new self(self::OTHER);

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
     * Return an array with all possible data types.
     *
     * @return list<self>
     */
    public static function all() : array
    {
        return array_map([self::class, 'build'], self::ALL);
    }

    /**
     * @param int $code
     * @throws InvalidArgumentException
     */
    public function __construct(int $code)
    {
        if (!in_array($code, self::ALL)) {
            throw new InvalidArgumentException('Invalid type code.');
        }

        $this->code = $code;
    }

    /**
     * Return the integer-encoded data type.
     *
     * @return int
     */
    public function code() : int
    {
        return $this->code;
    }

    /**
     * Is the data type continuous?
     *
     * @return bool
     */
    public function isContinuous() : bool
    {
        return $this->code === self::CONTINUOUS;
    }

    /**
     * Is the data type categorical?
     *
     * @return bool
     */
    public function isCategorical() : bool
    {
        return $this->code === self::CATEGORICAL;
    }

    /**
     * Is the data type an image resource?
     *
     * @return bool
     */
    public function isImage() : bool
    {
        return $this->code === self::IMAGE;
    }

    /**
     * Does the data not belong to any type?
     *
     * @return bool
     */
    public function isOther() : bool
    {
        return $this->code === self::OTHER;
    }

    /**
     * Return the data type as a string.
     *
     * @return string
     */
    public function __toString() : string
    {
        return self::TYPE_STRINGS[$this->code];
    }
}
