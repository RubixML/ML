<?php

namespace Rubix\ML;

use function call_user_func;

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
    protected $fn;

    /**
     * The arguments to the function.
     *
     * @var array
     */
    protected $args;

    /**
     * @param callable $fn
     * @param array $args
     */
    public function __construct(callable $fn, array $args = [])
    {
        $this->fn = $fn;
        $this->args = $args;
    }

    /**
     * Run the computation.
     *
     * @return mixed
     */
    public function compute()
    {
        return call_user_func($this->fn, ...$this->args);
    }

    /**
     * Invoke the object as a function.
     *
     * @return mixed
     */
    public function __invoke()
    {
        return $this->compute();
    }
}
