<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function is_string;
use function is_int;
use function is_float;

/**
 * Boolean Converter
 *
 * Convert truthy values to either categorical or continuous values.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Zach Vander Velden
 */
class BooleanConverter implements Transformer
{
    /**
     * The value used to replace boolean value `true` with.
     *
     * @var string|int|float
     */
    protected string|int|float $trueValue;

    /**
     * The value used to replace boolean value `false` with.
     *
     * @var string|int|float
     */
    protected string|int|float $falseValue;

    /**
     * @param string|int|float $trueValue
     * @param string|int|float $falseValue
     * @throws InvalidArgumentException
     */
    public function __construct(string|int|float $trueValue = 1, string|int|float $falseValue = 0)
    {
        if (is_string($trueValue) and !is_string($falseValue)) {
            throw new InvalidArgumentException('True and false values must'
                . ' be of the same data type.');
        }

        if (is_int($trueValue) and !is_int($falseValue)) {
            throw new InvalidArgumentException('True and false values must'
                . ' be of the same data type.');
        }

        if (is_float($trueValue) and !is_float($falseValue)) {
            throw new InvalidArgumentException('True and false values must'
                . ' be of the same data type.');
        }

        $this->trueValue = $trueValue;
        $this->falseValue = $falseValue;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Transform the dataset in place.
     *
     * @param array<mixed[]> $samples
     */
    public function transform(array &$samples) : void
    {
        array_walk($samples, [$this, 'convert']);
    }

    /**
     * Convert booleans to their user-defined values.
     *
     * @param list<mixed> $sample
     */
    public function convert(array &$sample) : void
    {
        foreach ($sample as &$value) {
            $value = $value ? $this->trueValue : $this->falseValue;
        }

        unset($value);
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Boolean Converter (true value: {$this->trueValue}, false value: {$this->falseValue})";
    }
}
