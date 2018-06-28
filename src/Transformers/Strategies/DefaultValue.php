<?php

namespace Rubix\ML\Transformers\Strategies;

use InvalidArgumentException;
use RuntimeException;

class DefaultValue implements Categorical, Continuous
{
    /**
     * The default value to impute.
     *
     * @var mixed
     */
    protected $value;

    /**
     * @param  mixed  $value
     * @return void
     */
    public function __construct($value)
    {
        if (!is_int($value) and !is_float($value) and !is_string($value)) {
            throw new InvalidArgumentException('Default value must be either a string, integer, or float.');
        }

        $this->value = $value;
    }

    /**
     * Needs no fiting.
     *
     * @param  array  $values
     * @return void
     */
    public function fit(array $values) : void
    {
        //
    }

    /**
     * Impute the default value.
     *
     * @return mixed
     */
    public function guess()
    {
        return $this->value;
    }
}
