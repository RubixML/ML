<?php

namespace Rubix\ML\Other\Strategies;

use Rubix\ML\DataType;

use function is_string;

/**
 * Constant
 *
 * Always guess the same value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Constant implements Strategy
{
    /**
     * The constant value to guess.
     *
     * @var string|int|float
     */
    protected $value;

    /**
     * @param string|int|float $value
     */
    public function __construct($value = 0)
    {
        $this->value = $value;
    }

    /**
     * Return the data type the strategy handles.
     *
     * @return \Rubix\ML\DataType
     */
    public function type() : DataType
    {
        return is_string($this->value)
            ? DataType::categorical()
            : DataType::continuous();
    }

    /**
     * Has the strategy been fitted?
     *
     * @internal
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return true;
    }

    /**
     * Fit the guessing strategy to a set of values.
     *
     * @internal
     *
     * @param list<string|int|float> $values
     */
    public function fit(array $values) : void
    {
        //
    }

    /**
     * Make a guess.
     *
     * @internal
     *
     * @return string|int|float
     */
    public function guess()
    {
        return $this->value;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Constant (value: {$this->value})";
    }
}
