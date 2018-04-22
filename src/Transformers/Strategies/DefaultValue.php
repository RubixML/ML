<?php

namespace Rubix\Engine\Transformers\Strategies;

use InvalidArgumentException;
use RuntimeException;

class DefaultValue implements Categorical, Continuous
{
    /**
     * The default class for categorical columns.
     *
     * @var string
     */
    protected $class;

    /**
     * The default value for continuous columns.
     *
     * @var string
     */
    protected $value;

    /**
     * @param  string  $categorical
     * @param  mixed  $continuous
     * @return void
     */
    public function __construct(string $class = 'unknown', $value = 0.0)
    {
        if (!is_int($value) && !is_float($value)) {
            throw new InvalidArgumentException('Default value must be an integer or float.');
        }

        $this->class = $class;
        $this->value = $value;
    }

    /**
     * Guess the value by referring to the default value for the column type.
     *
     * @param  array  $values
     * @return mixed
     */
    public function guess(array $values)
    {
        if (empty($values)) {
            throw new RuntimeException('This strategy requires at least 1 data point.');
        }

        return is_string(reset($values)) ? $this->class : $this->value;
    }
}
