<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

/**
 * Split
 *
 * Split nodes represent the value and index at which a dataset is
 * split to form left and right subgroups.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Split extends BinaryNode
{
    /**
     * The feature column (index) of the split value.
     *
     * @var int
     */
    protected $column;

    /**
     * The value that the node splits on.
     *
     * @var int|float|string
     */
    protected $value;

    /**
     * The left and right splits of the training data.
     *
     * @var array
     */
    protected $groups;

    /**
     * @param  int  $column
     * @param  mixed  $value
     * @param  array  $groups
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $column, $value, array $groups)
    {
        if (!is_string($value) and !is_numeric($value)) {
            throw new InvalidArgumentException('Split value must be a string'
                . ' or numeric type, ' . gettype($value) . ' found.');
        }

        if (count($groups) !== 2) {
            throw new InvalidArgumentException('The number of sample groups'
                . ' must be exactly 2.');
        }

        foreach ($groups as $group) {
            if (!$group instanceof Dataset) {
                throw new InvalidArgumentException('Sample groups must be'
                    . ' dataset objects, ' . gettype($group) . ' found.');
            }
        }

        $this->column = $column;
        $this->value = $value;
        $this->groups = $groups;
    }

    /**
     * Return the feature column (index) of the split value.
     *
     * @return int
     */
    public function column() : int
    {
        return $this->column;
    }

    /**
     * Return the split value.
     *
     * @return int|float|string
     */
    public function value()
    {
        return $this->value;
    }

    /**
     * Return the left and right splits of the training data.
     *
     * @return array
     */
    public function groups() : array
    {
        return $this->groups;
    }

    /**
     * Remove the left and right splits of the training data.
     *
     * @return void
     */
    public function cleanup() : void
    {
        unset($this->groups);
    }
}
