<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;

/**
 * Neighborhood
 *
 * Neighborhoods represent a group of samples that are close to
 * each other in distsance but not necessarily the closest.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Neighborhood extends BinaryNode implements Spatial, Leaf
{
    /**
     * The samples that make up the neighborhood.
     *
     * @var array
     */
    protected $samples;

    /**
     * The labels that make up the neighborhood.
     *
     * @var array
     */
    protected $labels;

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
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return void
     */
    public function __construct(Labeled $dataset)
    {
        $min = $max = [];

        foreach ($dataset->columns() as $values) {
            $min[] = min($values);
            $max[] = max($values);
        }

        $this->min = $min;
        $this->max = $max;
        $this->samples = $dataset->samples();
        $this->labels = $dataset->labels();
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

    /**
     * Return a tuple of the samples and labels stored in the neighborhood.
     * 
     * @return array[]
     */
    public function neighbors() : array
    {
        return [$this->samples, $this->labels];
    }

    /**
     * Return the samples in the neighborhood.
     * 
     * @return array[]
     */
    public function samples() : array
    {
        return $this->samples;
    }

    /**
     * Return the labels cooresponding to the samples in the neighborhood.
     * 
     * @return (int|float|string)[]
     */
    public function labels() : array
    {
        return $this->labels;
    }

    /**
     * Return the number of samples that this neighborhood holds.
     * 
     * @return int
     */
    public function n() : int
    {
        return count($this->samples);
    }
}
