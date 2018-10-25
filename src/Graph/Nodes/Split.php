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
     * The value that the node splits on.
     *
     * @var int|float|string
     */
    protected $value;

    /**
     * The feature column (index) of the split value.
     *
     * @var int
     */
    protected $index;

    /**
     * The left and right splits of the training data.
     *
     * @var array
     */
    protected $groups;

    /**
     * @param  mixed  $value
     * @param  int  $index
     * @param  array  $groups
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($value, int $index, array $groups)
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

        $this->value = $value;
        $this->index = $index;
        $this->groups = $groups;
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
     * Return the feature column (index) of the split value.
     * 
     * @return int
     */
    public function index() : int
    {
        return $this->index;
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
