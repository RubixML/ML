<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Other\Traits\RankSingle;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
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
class PersistentModel implements Estimator, Learner, Wrapper, Probabilistic, Ranking
{
    use PredictsSingle, ProbaSingle, RankSingle;
    
    /**
     * The persistable learner.
     *
     * @var \Rubix\ML\Learner
     */
    protected $estimator;

    /**
     * The persister object used to interface with the storage medium.
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
            throw new InvalidArgumentException('Persistable must'
                . ' implement the Learner interface.');
        }

        return new self($learner, $persister);
    }

    /**
     * @param \Rubix\ML\Learner $estimator
     * @param \Rubix\ML\Persisters\Persister $persister
     * @throws \InvalidArgumentException
     */
    public function __construct(Learner $estimator, Persister $persister)
    {
        if (!$estimator instanceof Persistable) {
            throw new InvalidArgumentException('Base learner must'
                . ' implement the Persistable interface.');
        }

        $this->estimator = $estimator;
        $this->persister = $persister;
    }

    /**
     * Return the estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return $this->estimator->type();
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->estimator->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'estimator' => $this->estimator,
            'persister' => $this->persister,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->estimator->trained();
    }

    /**
     * Return the base estimator instance.
     *
     * @return \Rubix\ML\Estimator
     */
    public function base() : Estimator
    {
        return $this->estimator;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        $this->estimator->train($dataset);
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->estimator->predict($dataset);
    }

    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->estimator instanceof Probabilistic) {
            throw new RuntimeException('Base Estimator must'
                . ' implement the Probabilistic interface.');
        }

        return $this->estimator->proba($dataset);
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return float[]
     */
    public function rank(Dataset $dataset) : array
    {
        if (!$this->estimator instanceof Ranking) {
            throw new RuntimeException('Base Estimator must'
                . ' implement the Ranking interface.');
        }
            
        return $this->estimator->rank($dataset);
    }

    /**
     * Save the model using the user-provided persister.
     */
    public function save() : void
    {
        if ($this->estimator instanceof Persistable) {
            $this->persister->save($this->estimator);
        }
    }

    /**
     * Allow methods to be called on the model from the wrapper.
     *
     * @param string $name
     * @param mixed[] $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        return $this->estimator->$name(...$arguments);
    }
}
