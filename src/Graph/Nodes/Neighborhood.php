<?php

namespace Rubix\ML\Graph\Nodes;

class Neighborhood extends BinaryNode
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
     * Return the samples.
     *
     * @return mixed
     */
    public function samples()
    {
        return $this->samples;
    }

    /**
     * Return the labels.
     *
     * @return mixed
     */
    public function labels()
    {
        return $this->labels;
    }
}
