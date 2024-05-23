<?php

namespace Rubix\ML;

use Rubix\ML\Helpers\Params;
use Rubix\ML\Serializers\RBX;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Serializers\Serializer;
use Rubix\ML\AnomalyDetectors\Scoring;
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
class PersistentModel implements EstimatorWrapper, Learner, Probabilistic, Scoring
{
    /**
     * The persistable base learner.
     *
     * @var Learner
     */
    protected Learner $base;

    /**
     * The persister used to interface with the storage layer.
     *
     * @var Persister
     */
    protected Persister $persister;

    /**
     * The object serializer.
     *
     * @var Serializer
     */
    protected Serializer $serializer;

    /**
     * Factory method to restore the model from persistence.
     *
     * @param Persister $persister
     * @param Serializer|null $serializer
     * @throws InvalidArgumentException
     * @return self
     */
    public static function load(Persister $persister, ?Serializer $serializer = null) : self
    {
        $serializer = $serializer ?? new RBX();

        $base = $serializer->deserialize($persister->load());

        if (!$base instanceof Learner) {
            throw new InvalidArgumentException('Persistable must'
                . ' implement the Learner interface.');
        }

        return new self($base, $persister, $serializer);
    }

    /**
     * @param Learner $base
     * @param Persister $persister
     * @param Serializer|null $serializer
     * @throws InvalidArgumentException
     */
    public function __construct(Learner $base, Persister $persister, ?Serializer $serializer = null)
    {
        if (!$base instanceof Persistable) {
            throw new InvalidArgumentException('Base Learner must'
                . ' implement the Persistable interface.');
        }

        $this->base = $base;
        $this->persister = $persister;
        $this->serializer = $serializer ?? new RBX();
    }

    /**
     * Return the estimator type.
     *
     * @internal
     *
     * @return EstimatorType
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
            'serializer' => $this->serializer,
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
     * @return Estimator
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
        if (!$this->base instanceof Persistable) {
            throw new RuntimeException('Base estimator is not persistable.');
        }

        $encoding = $this->serializer->serialize($this->base);

        $this->persister->save($encoding);
    }

    /**
     * Train the learner with a dataset.
     *
     * @param Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        $this->base->train($dataset);
    }

    /**
     * Make a prediction on a given sample dataset.
     *
     * @param Dataset $dataset
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        return $this->base->predict($dataset);
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return list<float[]>
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
     * @param Dataset $dataset
     * @throws RuntimeException
     * @return float[]
     */
    public function score(Dataset $dataset) : array
    {
        if (!$this->base instanceof Scoring) {
            throw new RuntimeException('Base Estimator must'
                . ' implement the Scoring interface.');
        }

        return $this->base->score($dataset);
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
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Persistent Model (' . Params::stringify($this->params()) . ')';
    }
}
