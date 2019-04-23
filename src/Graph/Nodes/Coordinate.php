<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;
use Generator;

use function Rubix\ML\argmax;

/**
 * Coordinate
 *
 * A coordinate node represents a coordinate column of a k-d tree.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Coordinate extends BinaryNode implements Box
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
     * @var \Rubix\ML\Datasets\Labeled[]
     */
    protected $groups;

    /**
     * The minimum vector that encompasses all samples contained within.
     *
     * @var (int|float)[]
     */
    protected $min;

    /**
     * The maximum vector that encompasses all samples contained within.
     *
     * @var (int|float)[]
     */
    protected $max;

    /**
     * Factory method to build a coordinate node from a labeled dataset
     * using the column with the highest variance as the column for the
     * split.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return self
     */
    public static function split(Labeled $dataset) : self
    {
        $columns = $dataset->columns();

        $variances = array_map([Stats::class, 'variance'], $columns);

        $column = argmax($variances);

        $value = Stats::median($columns[$column]);

        $groups = $dataset->partition($column, $value);

        $min = $max = [];

        foreach ($columns as $values) {
            $min[] = min($values);
            $max[] = max($values);
        }

        return new self($column, $value, $groups, $min, $max);
    }

    /**
     * @param int $column
     * @param mixed $value
     * @param array $groups
     * @param array $min
     * @param array $max
     * @throws \InvalidArgumentException
     */
    public function __construct(int $column, $value, array $groups, array $min, array $max)
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
            if (!$group instanceof Labeled) {
                throw new InvalidArgumentException('Sample groups must be'
                    . ' dataset objects, ' . gettype($group) . ' given.');
            }
        }

        if (empty($min)) {
            throw new InvalidArgumentException('Bounding box cannot be empty');
        }

        if (count($min) !== count($max)) {
            throw new InvalidArgumentException('Min and max vectors must be'
                . ' the same dimensionality.');
        }

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
     * @return array
     */
    public function groups() : array
    {
        return $this->groups;
    }

    /**
     * Return the bounding box surrounding this node.
     *
     * @return \Generator
     */
    public function sides() : Generator
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
