<?php

namespace Rubix\Engine;

use Rubix\Engine\Persisters\Persister;
use Rubix\Engine\Persisters\Persistable;
use InvalidArgumentException;

class PersistentModel
{
    /**
     * The estimator.
     *
     * @var \Rubix\Engine\Estimators\Estimator
     */
    protected $estimator;

    /**
     * The connector responsible for persisting the model.
     *
     * @var \Rubix\Engine\Persisters\Persister
     */
    protected $persister;

    /**
     * Factory method to restore the model from persistence.
     *
     * @param \Rubix\Engine\Persisters\Persister  $persister
     * @return self
     */
    public static function restore(Persister $persister) : self
    {
        return new self($persister->restore(), $persister);
    }

    /**
     * @param  \Rubix\Engine\Persisters\Persistable  $model
     * @param  \Rubix\Engine\Persisters\Persister  $persister
     * @return void
     */
    public function __construct(Persistable $model, Persister $persister)
    {
        $this->model = $model;
        $this->connector = $persister;
    }

    /**
     * Return the underlying model instance.
     *
     * @return \Rubix\Engine\Persisters\Persistable
     */
    public function model() : Persistable
    {
        return $this->model;
    }

    /**
     * @return \Rubix\Engine\Persisters\Persister
     */
    public function persister() : Persister
    {
        return $this->persister;
    }

    /**
     * Save the model to storage.
     *
     * @return void
     */
    public function save() : bool
    {
        return $this->persister->save($this->model);
    }

    /**
     * Allow methods to be called from the base model from the persistent model.
     *
     * @param  string  $name
     * @param  array  $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        return $this->model->$name(...$arguments);
    }
}
