<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function is_numeric;
use function is_string;
use function is_bool;

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
     * The value used to replace boolean value `true` with.
     *
     * @var string|float|int
     */
    protected $trueValue;

    /**
     * The value used to replace boolean value `false` with.
     *
     * @var string|float|int
     */
    protected $falseValue;

    /**
     * @param mixed $trueValue
     * @param mixed $falseValue
     */
    public function __construct($trueValue = 'true', $falseValue = 'false')
    {
        if (!is_numeric($trueValue) && !is_string($trueValue)) {
            throw new InvalidArgumentException('True value must be'
                . ' a string or numeric type.');
        }

        if (!is_numeric($falseValue) && !is_string($falseValue)) {
            throw new InvalidArgumentException('False value must be'
                . ' a string or numeric type.');
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
        return "Boolean Converter (true: {$this->trueValue}, false: {$this->falseValue})";
    }
}
