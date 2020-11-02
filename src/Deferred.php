<?php

namespace Rubix\ML;

use function call_user_func_array;

/**
 * Deferred
 *
 * A deferred computation i.e. an object that represents the result of a computation
 * performed sometime in the future.
 *
 * @internal
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
     * @var mixed[]
     */
    protected $args;

    /**
     * @param callable $fn
     * @param mixed[] $args
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
        return call_user_func_array($this->fn, $this->args);
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
