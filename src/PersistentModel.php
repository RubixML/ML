<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Other\Traits\LoggerAware;
use InvalidArgumentException;
use RuntimeException;

/**
 * Persistent Model
 *
 * The Persistent Model wrapper gives the estimator two additional methods (`save()`
 * and `load()`) that allow the estimator to be saved and retrieved from storage.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PersistentModel implements Estimator, Learner, Wrapper, Probabilistic, Verbose
{
    use LoggerAware;

    /**
     * The underlying persistable estimator instance.
     *
     * @var \Rubix\ML\Learner
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
     * @param \Rubix\ML\Persisters\Persister $persister
     * @throws \InvalidArgumentException
     * @return self
     */
    public static function load(Persister $persister) : self
    {
        $learner = $persister->load();

        if (!$learner instanceof Learner) {
            throw new InvalidArgumentException('Peristable must be a'
                . ' learner.');
        }

        return new self($learner, $persister);
    }

    /**
     * @param \Rubix\ML\Learner $base
     * @param \Rubix\ML\Persisters\Persister $persister
     * @throws \InvalidArgumentException
     */
    public function __construct(Learner $base, Persister $persister)
    {
        if (!$base instanceof Persistable) {
            throw new InvalidArgumentException('Base estimator must'
                . ' be persistable.');
        }

        $this->base = $base;
        $this->persister = $persister;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->base->type();
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->base->compatibility();
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->base->trained();
    }

    /**
     * Return the base estimator instance.
     *
     * @return \Rubix\ML\Estimator
     */
    public function base() : Estimator
    {
        return $this->base;
    }

    /**
     * Train the underlying estimator.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        $this->base->train($dataset);
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->base->predict($dataset);
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $base = $this->base();

        if ($base instanceof Probabilistic) {
            return $base->proba($dataset);
        }

        throw new RuntimeException('Base estimator must'
            . ' implement the probabilistic interface.');
    }

    /**
     * Save the model using the user-provided persister.
     */
    public function save() : void
    {
        if ($this->base instanceof Persistable) {
            $this->persister->save($this->base);

            if ($this->logger) {
                $this->logger->info('Model saved successully');
            }
        }
    }

    /**
     * Allow methods to be called on the model from the wrapper.
     *
     * @param string $name
     * @param array $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        return $this->base->$name(...$arguments);
    }
}
