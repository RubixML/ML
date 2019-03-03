<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Other\Functions\Argmax;

/**
 * Coordinate
 *
 * A coordinate node represents a coordinate column of a k-d tree.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Coordinate extends Split implements BoundingBox
{
    /**
     * The multivariate minimum of the bounding box around the samples
     * in the neighborhood.
     *
     * @var array
     */
    protected $min;

    /**
     * The multivariate maximum of the bounding box around the samples
     * in the neighborhood.
     *
     * @var array
     */
    protected $max;

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function __construct(Dataset $dataset)
    {
        $columns = $dataset->columns();

        $variances = array_map([Stats::class, 'variance'], $columns);

        $column = Argmax::compute($variances);

        $value = Stats::median($columns[$column]);

        $min = $max = [];

        foreach ($columns as $values) {
            $min[] = min($values);
            $max[] = max($values);
        }

        $groups = $dataset->partition($column, $value);

        $this->min = $min;
        $this->max = $max;

        parent::__construct($column, $value, $groups);
    }

    /**
     * Return the bounding box surrounding this node.
     *
     * @return array[]
     */
    public function box() : array
    {
        return [$this->min, $this->max];
    }
}
