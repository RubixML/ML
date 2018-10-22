<?php

namespace Rubix\ML;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Persisters\Persister;
use InvalidArgumentException;
use RuntimeException;

/**
 * Persistent Model
 *
 * It is possible to persist a model to disk by wrapping the Estimator instance
 * in a Persistent Model meta-Estimator. The Persistent Model class gives the
 * Estimator two additional methods save() and restore() that serialize and
 * unserialize to and from disk. In order to be persisted the Estimator must
 * implement the Persistable interface.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PersistentModel implements MetaEstimator, Learner
{
    /**
     * The underlying persistable estimator instance.
     *
     * @var \Rubix\ML\Persistable
     */
    protected $base;

    /**
     * The persister responsible to saveing and restoring the estimator.
     *
     * @var \Rubix\ML\Persisters\Persister
     */
    protected $persister;

    /**
     * Factory method to restore the model from a persister.
     *
     * @param  \Rubix\ML\Persisters\Persister  $persister
     * @return self
     */
    public static function load(Persister $persister) : self
    {
        return new self($persister->load(), $persister);
    }

    /**
     * @param  \Rubix\ML\Persistable  $base
     * @param  \Rubix\ML\Persisters\Persister  $persister
     * @return void
     */
    public function __construct(Persistable $base, Persister $persister)
    {
        $this->base = $base;;
        $this->persister = $persister;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->base->type();
    }

    /**
     * Return the instance of the base estimator.
     *
     * @return \Rubix\ML\Persistable
     */
    public function estimator() : Persistable
    {
        return $this->base;
    }

    /**
     * Train the underlying estimator.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        $this->base->train($dataset);
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->base->predict($dataset);
    }

    /**
     * Save the model using the user-provided persister.
     *
     * @return void
     */
    public function save() : void
    {
        $this->persister->save($this->base);
    }

    /**
     * Delete the model from persistence.
     *
     * @return void
     */
    public function delete() : void
    {
        $this->persister->delete();
    }

    /**
     * Allow methods to be called on the model from the wrapper.
     *
     * @param  string  $name
     * @param  array  $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        return $this->base->$name(...$arguments);
    }
}
