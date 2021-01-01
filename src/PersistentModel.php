<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\AnomalyDetectors\Scoring;
use Rubix\ML\Other\Traits\RanksSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

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
class PersistentModel implements Estimator, Learner, Wrapper, Probabilistic, Scoring, Ranking
{
    use PredictsSingle, ProbaSingle, RanksSingle;

    /**
     * The persistable base learner.
     *
     * @var \Rubix\ML\Learner
     */
    protected $base;

    /**
     * The persister used to interface with the storage medium.
     *
     * @var \Rubix\ML\Persisters\Persister
     */
    protected $persister;

    /**
     * Factory method to restore the model from persistence.
     *
     * @param \Rubix\ML\Persisters\Persister $persister
     * @return self
     */
    public static function load(Persister $persister) : self
    {
        $base = $persister->load();

        if (!$base instanceof Learner) {
            throw new InvalidArgumentException('Persistable must'
                . ' implement the Learner interface.');
        }

        return new self($base, $persister);
    }

    /**
     * @param \Rubix\ML\Learner $base
     * @param \Rubix\ML\Persisters\Persister $persister
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(Learner $base, Persister $persister)
    {
        if (!$base instanceof Persistable) {
            throw new InvalidArgumentException('Base Learner must'
                . ' implement the Persistable interface.');
        }

        $this->base = $base;
        $this->persister = $persister;
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return $this->base->type();
    }

    /**
     * Return the data types that the estimator is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return $this->base->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @internal
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'base' => $this->base,
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
     * Save the model to storage.
     */
    public function save() : void
    {
        if ($this->base instanceof Persistable) {
            $this->persister->save($this->base);
        }
    }

    /**
     * Train the learner with a dataset.
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
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->base->predict($dataset);
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        if (!$this->base instanceof Probabilistic) {
            throw new RuntimeException('Base Estimator must'
                . ' implement the Probabilistic interface.');
        }

        return $this->base->proba($dataset);
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return float[]
     */
    public function score(Dataset $dataset) : array
    {
        if (!$this->base instanceof Scoring) {
            throw new RuntimeException('Base Estimator must'
                . ' implement the Ranking interface.');
        }

        return $this->base->score($dataset);
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @deprecated
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return float[]
     */
    public function rank(Dataset $dataset) : array
    {
        warn_deprecated('Rank() is deprecated, use score() instead.');

        return $this->score($dataset);
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
        return $this->base->$name(...$arguments);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Persistent Model (' . Params::stringify($this->params()) . ')';
    }
}
