<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;

class Neighborhood extends BinaryNode
{
    /**
     * A partitioned dataset that makes up the neighborhood of the node.
     *
     * @var \Rubix\ML\Datasets\Labeled
     */
    protected $dataset;

    /**
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @return void
     */
    public function __construct(Labeled $dataset)
    {
        $this->dataset = $dataset;
    }

    /**
     * Return the dataset.
     *
     * @return mixed
     */
    public function dataset()
    {
        return $this->dataset;
    }
}
