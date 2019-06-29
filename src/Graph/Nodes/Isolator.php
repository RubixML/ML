<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

/**
 * Isolator
 *
 * Isolator nodes represent splits in a tree designed to isolate groups
 * into cells by randomly dividing them.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Isolator extends BinaryNode
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
     * @var \Rubix\ML\Datasets\Dataset[]
     */
    protected $groups;

    /**
     * Factory method to build a isolator node from a dataset
     * using a random split of the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return self
     */
    public static function split(Dataset $dataset) : self
    {
        $column = rand(0, $dataset->numColumns() - 1);

        $value = $dataset[rand(0, count($dataset) - 1)][$column];

        $groups = $dataset->partition($column, $value);

        return new self($column, $value, $groups);
    }

    /**
     * @param int $column
     * @param mixed $value
     * @param array $groups
     * @throws \InvalidArgumentException
     */
    public function __construct(int $column, $value, array $groups)
    {
        if (!is_string($value) and !is_numeric($value)) {
            throw new InvalidArgumentException('Split value must be a string'
                . ' or numeric type, ' . gettype($value) . ' given.');
        }

        if (count($groups) !== 2) {
            throw new InvalidArgumentException('The number of groups'
                . ' must be exactly 2.');
        }

        foreach ($groups as $group) {
            if (!$group instanceof Dataset) {
                throw new InvalidArgumentException('Sample groups must be'
                    . ' dataset objects, ' . gettype($group) . ' given.');
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
     */
    public function cleanup() : void
    {
        unset($this->groups);
    }
}
