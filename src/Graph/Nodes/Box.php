<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildrenTrait;
use Rubix\ML\Exceptions\RuntimeException;
use Traversable;

use function Rubix\ML\argmax;
use function min;
use function max;

/**
 * Box
 *
 * A 1-dimensional split node with bounding box containing samples in both left and right
 * subtrees.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Box implements Hypercube, HasBinaryChildren
{
    use HasBinaryChildrenTrait;

    /**
     * The feature column (index) of the split value.
     *
     * @var int
     */
    protected int $column;

    /**
     * The value that the node splits on.
     *
     * @var int|float|string
     */
    protected $value;

    /**
     * The left and right subsets of the training data.
     *
     * @var array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled}
     */
    protected array $subsets;

    /**
     * The minimum vector containing all the points.
     *
     * @var list<int|float>
     */
    protected array $min;

    /**
     * The maximum vector containing all the points.
     *
     * @var list<int|float>
     */
    protected array $max;

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

        foreach ($dataset->features() as $values) {
            $mins[] = $min = min($values);
            $maxs[] = $max = max($values);

            $ranges[] = $max - $min;
        }

        $column = argmax($ranges);

        $value = 0.5 * ($mins[$column] + $maxs[$column]);

        $subsets = $dataset->splitByFeature($column, $value);

        return new self($column, $value, $subsets, $mins, $maxs);
    }

    /**
     * @param int $column
     * @param string|int|float $value
     * @param array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled} $subsets
     * @param list<int|float> $min
     * @param list<int|float> $max
     */
    public function __construct(int $column, $value, array $subsets, array $min, array $max)
    {
        $this->column = $column;
        $this->value = $value;
        $this->subsets = $subsets;
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
     * Return the left and right subsets of the training data.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return array{\Rubix\ML\Datasets\Labeled,\Rubix\ML\Datasets\Labeled}
     */
    public function subsets() : array
    {
        if (!isset($this->subsets)) {
            throw new RuntimeException('Subsets property does not exist.');
        }

        return $this->subsets;
    }

    /**
     * Return a generator with the bounding box surrounding this node.
     *
     * @return \Generator<list<int|float>>
     */
    public function sides() : Traversable
    {
        yield $this->min;
        yield $this->max;
    }

    /**
     * Does the hypercube reduce to a single point?
     *
     * @return bool
     */
    public function isPoint() : bool
    {
        return $this->min == $this->max;
    }

    /**
     * Remove any variables carried over from the parent node.
     */
    public function cleanup() : void
    {
        unset($this->subsets);
    }
}
