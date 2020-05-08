<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Persisters\Persister;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Other\Traits\RanksSingle;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Psr\Log\LoggerInterface;
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
class PersistentModel implements Estimator, Learner, Wrapper, Probabilistic, Ranking, Verbose
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
     * The PSR-3 logger instance.
     *
     * @var \Psr\Log\LoggerInterface|null
     */
    protected $logger;

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

        $estimator = new self($base, $persister);

        if ($base instanceof Verbose) {
            $logger = $base->logger();

            if (isset($logger)) {
                $logger->info('Model loaded from ' . Params::shortName(get_class($persister)));

                $estimator->setLogger($logger);
            }
        }

        return $estimator;
    }

    /**
     * @param \Rubix\ML\Learner $base
     * @param \Rubix\ML\Persisters\Persister $persister
     * @throws \InvalidArgumentException
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
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return $this->base->type();
    }

    /**
     * Return the data types that the model is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->base->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
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
     * Sets a logger instance on the object.
     *
     * @param \Psr\Log\LoggerInterface $logger
     */
    public function setLogger(LoggerInterface $logger) : void
    {
        if ($this->base instanceof Verbose) {
            $this->base->setLogger($logger);
        }

        $this->logger = $logger;
    }

    /**
     * Return the logger or null if not set.
     *
     * @return \Psr\Log\LoggerInterface|null
     */
    public function logger() : ?LoggerInterface
    {
        return $this->logger;
    }

    /**
     * Set the storage driver used to save the model.
     *
     * @param \Rubix\ML\Persisters\Persister $persister
     */
    public function setPersister(Persister $persister) : void
    {
        $this->persister = $persister;
    }

    /**
     * Save the model to storage.
     */
    public function save() : void
    {
        if ($this->base instanceof Persistable) {
            $this->persister->save($this->base);

            if ($this->logger) {
                $this->logger->info('Model saved to ' . Params::shortName(get_class($this->persister)));
            }
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
     * @throws \RuntimeException
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
     * @throws \RuntimeException
     * @return float[]
     */
    public function rank(Dataset $dataset) : array
    {
        if (!$this->base instanceof Ranking) {
            throw new RuntimeException('Base Estimator must'
                . ' implement the Ranking interface.');
        }
            
        return $this->base->rank($dataset);
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
}
