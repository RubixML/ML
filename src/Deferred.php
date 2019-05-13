<?php

namespace Rubix\ML;

use function is_null;

/**
 * Deferred
 *
 * A deferred computation i.e. an object that represents the result of a
 * computation performed sometime in the future.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Deferred
{
    /**
     * The function containing the computation.
     *
     * @var callable
     */
    protected $function;

    /**
     * The arguments to the function.
     *
     * @var array
     */
    protected $args;

    /**
     * The memoized result of the computation.
     *
     * @var mixed|null
     */
    protected $result;

    /**
     * @param callable $function
     * @param array $args
     */
    public function __construct(callable $function, array $args = [])
    {
        $this->function = $function;
        $this->args = $args;
    }

    /**
     * Run the computation.
     *
     * @return mixed
     */
    public function compute()
    {
        return ($this->function)(...$this->args);
    }

    /**
     * Return the result of the computation.
     *
     * @return mixed
     */
    public function result()
    {
        if (is_null($this->result)) {
            $this->result = $this->compute();
        }

        return $this->result;
    }
}
