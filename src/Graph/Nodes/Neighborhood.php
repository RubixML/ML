<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Traits\HasBinaryChildren;
use Traversable;

/**
 * Neighborhood
 *
 * Neighborhoods are hypercubes that contain samples that are close to each other in distance but
 * not *necessarily* the closest.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Neighborhood implements BinaryNode, Hypercube
{
    use HasBinaryChildren;

    /**
     * The dataset stored in the node.
     *
     * @var \Rubix\ML\Datasets\Labeled
     */
    protected $dataset;

    /**
     * The multivariate minimum of the bounding box.
     *
     * @var list<int|float>
     */
    protected $min;

    /**
     * The multivariate maximum of the bounding box.
     *
     * @var list<int|float>
     */
    protected $max;

    /**
     * Terminate a branch with a dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return self
     */
    public static function terminate(Labeled $dataset) : self
    {
        $min = $max = [];

        foreach ($dataset->columns() as $values) {
            $min[] = min($values);
            $max[] = max($values);
        }

        return new self($dataset, $min, $max);
    }

    /**
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param list<int|float> $min
     * @param list<int|float> $max
     */
    public function __construct(Labeled $dataset, array $min, array $max)
    {
        $this->dataset = $dataset;
        $this->min = $min;
        $this->max = $max;
    }

    /**
     * Return the dataset stored in the node.
     *
     * @return \Rubix\ML\Datasets\Labeled
     */
    public function dataset() : Labeled
    {
        return $this->dataset;
    }

    /**
     * Return the bounding box surrounding this node.
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
}
