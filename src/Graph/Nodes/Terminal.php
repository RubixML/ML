<?php

namespace Rubix\ML\Graph\Nodes;

class Terminal extends BinaryNode
{
    /**
     * The predicted outcome.
     *
     * @var mixed
     */
    protected $outcome;

    /**
     * The prediction meta data.
     *
     * @var array
     */
    protected $meta = [
        //
    ];

    /**
     * @param  mixed  $outcome
     * @param  array  $meta
     * @return void
     */
    public function __construct($outcome, array $meta = [])
    {
        $this->outcome = $outcome;
        $this->meta = $meta;
    }

    /**
     * Return the predicted outcome.
     *
     * @return mixed
     */
    public function outcome()
    {
        return $this->outcome;
    }

    /**
     * Return a meta value.
     *
     * @return mixed
     */
    public function meta(string $property)
    {
        return $this->meta[$property] ?? null;
    }
}
