<?php

namespace Rubix\ML\NeuralNet\Parameters;

use Tensor\Tensor;

abstract class Parameter
{
    /**
     * The unique identifier of this parameter.
     *
     * @var int
     */
    protected $id;

    /**
     * The class auto incrementing id.
     *
     * @var int
     */
    protected static $counter = 0;

    public function __construct()
    {
        $this->id = self::$counter++;
    }

    /**
     * Return the parameter tensor.
     *
     * @return mixed
     */
    abstract public function w();

    /**
     * Update the parameter tensor.
     *
     * @param \Tensor\Tensor $step
     */
    abstract public function update(Tensor $step) : void;

    /**
     * Return the unique identifier of the parameter.
     *
     * @return int
     */
    public function id() : int
    {
        return $this->id;
    }
}
