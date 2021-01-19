<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;

use const Rubix\ML\PHI;

/**
 * Isolator
 *
 * Isolator nodes represent splits in a tree designed to isolate groups into cells by randomly
 * dividing them.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Isolator implements BinaryNode
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
     * @var array{\Rubix\ML\Datasets\Dataset,\Rubix\ML\Datasets\Dataset}
     */
    protected $groups;

    /**
     * Factory method to build a isolator node from a dataset using a random split
     * of the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return self
     */
    public static function split(Dataset $dataset) : self
    {
        $column = rand(0, $dataset->numColumns() - 1);

        $values = $dataset->column($column);

        if ($dataset->columnType($column)->isContinuous()) {
            $min = (int) floor(min($values) * PHI);
            $max = (int) ceil(max($values) * PHI);

            $value = rand($min, $max) / PHI;
        } else {
            $offset = array_rand(array_unique($values));

            $value = $values[$offset];
        }

        $groups = $dataset->splitByColumn($column, $value);

        return new self($column, $value, $groups);
    }

    /**
     * @param int $column
     * @param string|int|float $value
     * @param array{Dataset,Dataset} $groups
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $column, $value, array $groups)
    {
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
     * @return array{\Rubix\ML\Datasets\Dataset,\Rubix\ML\Datasets\Dataset}
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
