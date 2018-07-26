<?php

namespace Rubix\ML\Graph\Nodes;

use InvalidArgumentException;

class Split extends BinaryNode
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
     * @param  mixed  $value
     * @param  array  $groups
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $index, $value, array $groups)
    {
        if (count($groups) !== 2) {
            throw new InvalidArgumentException('The number of groups must be'
                . ' exactly 2.');
        }

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
