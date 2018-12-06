<?php

namespace Rubix\ML;

use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Other\Traits\LoggerAware;

/**
 * Persistent Model
 *
 * It is possible to persist a model by wrapping the estimator instance
 * in a Persistent Model meta-estimator. The Persistent Model wrapper
 * gives the estimator three additional methods `save()`, `load()`, and
 * `prompt()` that allow the estimator to be saved and retrieved from
 * storage.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PersistentModel implements Learner, Verbose
{
    use LoggerAware;

    /**
     * The underlying persistable estimator instance.
     *
     * @var \Rubix\ML\Persistable
     */
    protected $base;

    /**
     * The persister is responsible for saving and restoring the estimator.
     *
     * @var \Rubix\ML\Persisters\Persister
     */
    protected $persister;

    /**
     * Factory method to restore the model from persistence.
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
        $this->base = $base;
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
     * Return the base estimator instance.
     *
     * @return \Rubix\ML\Estimator
     */
    public function estimator() : Estimator
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

        if ($this->logger) $this->logger->info('Model saved successully');
    }

    /**
     * Prompt the user to save this model or not.
     * 
     * @return void
     */
    public function prompt() : void
    {
        $save = strtolower(readline('Save this model? (y|[n]): '));

        if ($save === 'y') {
            $this->save();
        }
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
