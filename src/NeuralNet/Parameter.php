<?php

namespace Rubix\ML\NeuralNet;

use Tensor\Tensor;

/**
 * Parameter
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Parameter
{
    /**
     * The auto incrementing id.
     *
     * @var int
     */
    protected static $counter = 0;

    /**
     * The unique identifier of the parameter.
     *
     * @var int
     */
    protected $id;

    /**
     * The parameter.
     *
     * @var \Tensor\Tensor
     */
    protected $param;

    /**
     * @param \Tensor\Tensor $param
     */
    public function __construct(Tensor $param)
    {
        $this->id = self::$counter++;
        $this->param = $param;
    }

    /**
     * Return the unique identifier of the parameter.
     *
     * @return int
     */
    public function id() : int
    {
        return $this->id;
    }

    /**
     * Return the wrapped parameter.
     *
     * @return mixed
     */
    public function param()
    {
        return $this->param;
    }

    /**
     * Update the parameter.
     *
     * @param \Tensor\Tensor $step
     */
    public function update(Tensor $step) : void
    {
        $this->param = $this->param->subtract($step);
    }

    /**
     * Perform a deep copy of the object upon cloning.
     */
    public function __clone()
    {
        $this->param = clone $this->param;
    }
}
