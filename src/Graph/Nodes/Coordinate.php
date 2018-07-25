<?php

namespace Rubix\ML\Graph\Nodes;

class Coordinate extends BinaryNode
{
    /**
     * The index of the feature column.
     *
     * @var int
     */
    protected $index;

    /**
     * The split value.
     *
     * @var mixed
     */
    protected $value;

    /**
     * The left and right splits of the training data.
     *
     * @var array
     */
    protected $groups = [
        //
    ];

    /**
     * @param  int  $index
     * @param  float  $value
     * @param  array  $groups
     * @return void
     */
    public function __construct(int $index, float $value, array $groups)
    {
        $this->index = $index;
        $this->value = $value;
        $this->groups = $groups;
    }

    /**
     * @return int
     */
    public function index() : int
    {
        return $this->index;
    }

    /**
     * @return mixed
     */
    public function value()
    {
        return $this->value;
    }

    /**
     * @return array
     */
    public function groups() : array
    {
        return $this->groups;
    }

    /**
     * Remove the left and right splits of training data.
     *
     * @return void
     */
    public function cleanup() : void
    {
        unset($this->groups);
    }
}
