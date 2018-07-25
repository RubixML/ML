<?php

namespace Rubix\ML\Graph\Nodes;

class Isolator extends BinaryNode
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
     * The number of training samples this node is responsible for.
     *
     * @var int
     */
    protected $n;

    /**
     * @param  int  $index
     * @param  mixed  $value
     * @param  array  $groups
     * @return void
     */
    public function __construct(int $index, $value, array $groups)
    {
        $this->index = $index;
        $this->value = $value;
        $this->groups = $groups;
        $this->n = array_sum(array_map('count', $groups));
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
     * @return int
     */
    public function n() : int
    {
        return $this->n;
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
