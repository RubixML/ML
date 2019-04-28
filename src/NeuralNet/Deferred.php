<?php

namespace Rubix\ML\NeuralNet;

use Rubix\Tensor\Matrix;
use Closure;

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
     * The closure for the computation.
     *
     * @var \Closure
     */
    protected $computation;

    /**
     * The memoized result of the computation.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $result;

    /**
     * @param \Closure $computation
     */
    public function __construct(Closure $computation)
    {
        $this->computation = $computation;
    }

    /**
     * Return the result of the computation.
     *
     * @return \Rubix\Tensor\Matrix
     */
    public function result() : Matrix
    {
        if (!$this->result) {
            $this->result = call_user_func($this->computation);
        }

        return $this->result;
    }
}
