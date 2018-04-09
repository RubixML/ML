<?php

namespace Rubix\Engine;

use JsonSerializable;

class Prediction implements JsonSerializable
{
    /**
     * The outcome of a prediction.
     *
     * @var mixed
     */
    protected $outcome;

    /**
     * Any metadata that the estimator supplies along with the prediction.
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
     * @return mixed
     */
    public function outcome()
    {
        return $this->outcome;
    }

    /**
     * The output of the prediction is categorical.
     *
     * @return bool
     */
    public function categorical() : bool
    {
        return is_string($this->outcome);
    }

    /**
     * The output of the prediction is continuous.
     *
     * @return bool
     */
    public function continuous() : bool
    {
        return !$this->categorical() && is_numeric($this->outcome);
    }

    /**
     * Return a meta value by name or return the entire metadata array.
     *
     * @return mixed|null
     */
    public function meta(string $name = null)
    {
        if (!isset($name)) {
            return $this->meta;
        }

        return $this->meta[$name] ?? null;
    }

    /**
     * Add metadata to the prediction, overwrites previous meta values.
     *
     * @param  array  $meta
     * @return self
     */
    public function addMeta(array $meta) : self
    {
        $this->meta = array_replace($this->meta, $meta);

        return $this;
    }

    /**
     * @return array
     */
    public function toArray() : array
    {
        return [
            'outcome' => $this->outcome,
            'meta' => $this->meta,
        ];
    }

    /**
     * @return array
     */
    public function jsonSerialize()
    {
        return $this->toArray();
    }
}
