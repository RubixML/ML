<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Boolean Converter
 *
 * Convert boolean true/false values to continuous or categorical values.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Zach Vander Velden
 */
class BooleanConverter implements Transformer
{
    /**
     * The value used to replace boolean value `true` with. Use a string value to represent the boolean as a categorical
     * value. Use a float or int to represent the boolean as a continuous value.
     *
     * @var string|float|int
     */
    protected $trueValue;

    /**
     * The value used to replace boolean value `false` with. Use a string value to represent the boolean as a categorical
     * value. Use a float or int to represent the boolean as a continuous value.
     *
     * @var string|float|int
     */
    protected $falseValue;

    /**
     * BooleanConverter constructor.
     * @param mixed $trueValue
     * @param mixed $falseValue
     */
    public function __construct($trueValue = 'true', $falseValue = 'false')
    {
        if (!is_float($trueValue) && !is_int($trueValue) && !is_string($trueValue)) {
            throw new InvalidArgumentException('TrueValue must be'
                . ' a string, float, or int.');
        }

        if (!is_float($falseValue) && !is_int($falseValue) && !is_string($falseValue)) {
            throw new InvalidArgumentException('FalseValue must be'
                . ' a string, float, or int.');
        }

        $this->trueValue = $trueValue;
        $this->falseValue = $falseValue;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$value) {
                if (is_bool($value)) {
                    $value = $value
                        ? $this->trueValue
                        : $this->falseValue;
                }
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Boolean Converter';
    }
}
