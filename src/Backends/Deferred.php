<?php

namespace Rubix\ML\Backends;

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
     * Return the result of the computation.
     *
     * @return mixed
     */
    public function result()
    {
        if (!$this->result) {
            $this->result = ($this->function)(...$this->args);
        }

        return $this->result;
    }
}
