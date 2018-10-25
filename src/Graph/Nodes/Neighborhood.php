<?php

namespace Rubix\ML\Graph\Nodes;

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
     * The number of training samples this node is responsible for.
     *
     * @var int
     */
    protected $n;

    /**
     * @param  array  $samples
     * @param  array  $labels
     * @return void
     */
    public function __construct(array $samples, array $labels)
    {
        $this->samples = $samples;
        $this->labels = $labels;
        $this->n = count($samples);
    }

    /**
     * @return mixed
     */
    public function samples()
    {
        return $this->samples;
    }

    /**
     * @return mixed
     */
    public function labels()
    {
        return $this->labels;
    }

    /**
     * @return int
     */
    public function n() : int
    {
        return $this->n;
    }
}
