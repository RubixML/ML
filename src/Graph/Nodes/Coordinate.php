<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;

/**
 * Coordinate
 *
 * A coordinate node represents a coordinate column of a k-d tree.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Coordinate extends Split implements Spatial
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
     * @param  int  $column
     * @param  mixed  $value
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return void
     */
    public function __construct(int $column, $value, Labeled $dataset)
    {
        $min = $max = [];

        foreach ($dataset->rotate() as $values) {
            $min[] = min($values);
            $max[] = max($values);
        }

        $groups = $dataset->partition($column, $value);

        $this->min = $min;
        $this->max = $max;

        parent::__construct($column, $value, $groups);
    }

    /**
     * Return the bounding box around this node.
     * 
     * @return array[]
     */
    public function box() : array
    {
        return [$this->min, $this->max];
    }
}
