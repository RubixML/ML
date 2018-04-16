<?php

namespace Rubix\Engine;

use Rubix\Engine\Connectors\Connector;
use Rubix\Engine\Connectors\Persistable;
use InvalidArgumentException;

class PersistentModel implements Estimator
{
    /**
     * The estimator.
     *
     * @var \Rubix\Engine\Estimator
     */
    protected $estimator;

    /**
     * The connector responsible for persisting the model.
     *
     * @var \Rubix\Engine\Connectors\Connector
     */
    protected $connector;

    /**
     * Factory method to restore the model from persistence.
     *
     * @param \Rubix\Engine\Connectors\Connector  $connector
     * @return self
     */
    public static function restore(Connector $connector) : self
    {
        return new self($connector->restore(), $connector);
    }

    /**
     * @param  \Rubix\Engine\Estimator  $estimator
     * @param  \Rubix\Engine\Connector  $connector
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(Estimator $estimator, Connector $connector)
    {
        if (!$estimator instanceof Persistable) {
            throw new InvalidArgumentException('Estimator must implement the Persistable interface.');
        }

        $this->estimator = $estimator;
        $this->connector = $connector;
    }

    /**
     * Return the underlying estimator instance.
     *
     * @return \Rubix\Engine\Estimator
     */
    public function estimator() : Estimator
    {
        return $this->estimator;
    }

    /**
     * @return \Rubix\Engine\Connectors\Connector
     */
    public function connector() : Estimator
    {
        return $this->connector;
    }

    /**
     * Train the estimator.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function train(Dataset $data) : void
    {
        $this->estimator->train($data);
    }

    /**
     * Make a prediction of a given sample.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        return $this->estimator->predict($sample);
    }

    /**
     * Persist the model.
     *
     * @return void
     */
    public function save() : bool
    {
        return $this->connector->save($this->estimator);
    }
}
