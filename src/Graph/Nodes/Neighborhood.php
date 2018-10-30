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
class Neighborhood extends BinaryNode implements Leaf
{
    /**
     * The samples that make up the neighborhood.
     *
     * @var array
     */
    protected $samples;

    /**
     * The samples that make up the neighborhood.
     *
     * @var array
     */
    protected $labels;

    /**
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return void
     */
    public function __construct(Labeled $dataset)
    {
        $this->samples = $dataset->samples();
        $this->labels = $dataset->labels();
    }

    /**
     * Return the samples in the neighborhood.
     * 
     * @return array[]
     */
    public function samples()
    {
        return $this->samples;
    }

    /**
     * Return the labels cooresponding to the samples in the neighborhood.
     * 
     * @return (int|float|string)[]
     */
    public function labels()
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
