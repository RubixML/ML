<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;
use Traversable;

use function Rubix\ML\argmax;

/**
 * Box
 *
 * A 1-dimensional split node with bounding box containing samples in both left and right
 * subtrees.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Box implements BinaryNode, Hypercube
{
    use HasBinaryChildren;

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
     * @var \Rubix\ML\Datasets\Labeled[]
     */
    protected $groups;

    /**
     * The minimum vector containing all points.
     *
     * @var (int|float)[]
     */
    protected $min;

    /**
     * The maximum vector containing all points.
     *
     * @var (int|float)[]
     */
    protected $max;

    /**
     * Factory method to build a coordinate node from a labeled dataset
     * using the column with the highest range as the column for the split.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return self
     */
    public static function split(Labeled $dataset) : self
    {
        $mins = $maxs = $ranges = [];

        foreach ($dataset->columns() as $values) {
            $mins[] = $min = min($values);
            $maxs[] = $max = max($values);

            $ranges[] = $max - $min;
        }

        $column = argmax($ranges);

        $value = 0.5 * ($mins[$column] + $maxs[$column]);

        $groups = $dataset->partitionByColumn($column, $value);

        return new self($column, $value, $groups, $mins, $maxs);
    }

    /**
     * @param int $column
     * @param string|int|float $value
     * @param \Rubix\ML\Datasets\Labeled[] $groups
     * @param (int|float)[] $min
     * @param (int|float)[] $max
     */
    public function __construct(int $column, $value, array $groups, array $min, array $max)
    {
        $this->column = $column;
        $this->value = $value;
        $this->groups = $groups;
        $this->min = $min;
        $this->max = $max;
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
     * @return \Rubix\ML\Datasets\Labeled[]
     */
    public function groups() : array
    {
        return $this->groups;
    }

    /**
     * Return a generator with the bounding box surrounding this node.
     *
     * @return \Traversable<array>
     */
    public function sides() : Traversable
    {
        yield $this->min;
        yield $this->max;
    }

    /**
     * Remove the left and right splits of the training data.
     */
    public function cleanup() : void
    {
        unset($this->groups);
    }
}
