<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Transformers\Elastic;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Other\Traits\RankSingle;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Other\Traits\ProbaSingle;
use Rubix\ML\Other\Traits\PredictsSingle;
use Psr\Log\LoggerInterface;
use InvalidArgumentException;
use RuntimeException;

use function get_class;

/**
 * Pipeline
 *
 * Pipeline is a meta-estimator capable of transforming an input dataset by applying a
 * series of Transformer *middleware*. Under the hood, Pipeline will automatically fit the
 * training set and transform any Dataset object supplied as an argument to one of the base
 * estimator's methods before hitting the method context. With *elastic* mode enabled,
 * Pipeline will update the fitting of Elastic transformers during partial training.
 *
 * > **Note:** Since transformations are applied to dataset objects in-place (without making a
 * copy of the data), using a dataset in a program after it has been run through Pipeline may
 * have unexpected results. If you need to keep a *clean* dataset in memory you can clone
 * the dataset object before calling the method on Pipeline that consumes it.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Pipeline implements Online, Wrapper, Probabilistic, Ranking, Persistable, Verbose
{
    use PredictsSingle, ProbaSingle, RankSingle;

    /**
     * A list of transformers to be applied in order.
     *
     * @var \Rubix\ML\Transformers\Transformer[]
     */
    protected $transformers = [
        //
    ];

    /**
     * An instance of a base estimator to receive the transformed data.
     *
     * @var \Rubix\ML\Estimator
     */
    protected $estimator;

    /**
     * Should we update the elastic transformers during partial train?
     *
     * @var bool
     */
    protected $elastic;

    /**
     * The PSR-3 logger instance.
     *
     * @var \Psr\Log\LoggerInterface|null
     */
    protected $logger;

    /**
     * Whether or not the transformer pipeline has been fitted.
     *
     * @var bool
     */
    protected $fitted;

    /**
     * @param \Rubix\ML\Transformers\Transformer[] $transformers
     * @param \Rubix\ML\Estimator $estimator
     * @param bool $elastic
     * @throws \InvalidArgumentException
     */
    public function __construct(array $transformers, Estimator $estimator, bool $elastic = true)
    {
        foreach ($transformers as $transformer) {
            if (!$transformer instanceof Transformer) {
                throw new InvalidArgumentException('Transformer must'
                    . ' implement the Transformer interface.');
            }
        }

        $this->transformers = $transformers;
        $this->estimator = $estimator;
        $this->elastic = $elastic;
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
            'transformers' => $this->transformers,
            'estimator' => $this->estimator,
            'elastic' => $this->elastic,
        ];
    }

    /**
     * Sets a logger instance on the object.
     *
     * @param \Psr\Log\LoggerInterface $logger
     */
    public function setLogger(LoggerInterface $logger) : void
    {
        if ($this->estimator instanceof Verbose) {
            $this->estimator->setLogger($logger);
        }

        $this->logger = $logger;
    }

    /**
     * Return if the logger is logging or not.
     *
     * @var bool
     */
    public function logging() : bool
    {
        return isset($this->logger);
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->estimator instanceof Learner
            ? $this->estimator->trained()
            : true;
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
     * Run the training dataset through all transformers in order and use the
     * transformed dataset to train the estimator.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        $this->fit($dataset);

        if ($this->estimator instanceof Learner) {
            $this->estimator->train($dataset);
        }
    }

    /**
     * Perform a partial train.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        if ($this->elastic) {
            $this->update($dataset);
        }

        if ($this->estimator instanceof Online) {
            $this->estimator->partial($dataset);
        }
    }

    /**
     * Fit the transformer pipeline to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function fit(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Stateful) {
                $transformer->fit($dataset);

                if ($this->logger) {
                    $this->logger->info('Fitted ' . Params::shortName(get_class($transformer)));
                }
            }

            $dataset->apply($transformer);
        }
    }

    /**
     * Update the fittings of the transformers.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function update(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Elastic) {
                $transformer->update($dataset);

                if ($this->logger) {
                    $this->logger->info('Updated ' . Params::shortName(get_class($transformer)));
                }
            }

            $dataset->apply($transformer);
        }
    }

    /**
     * Preprocess the dataset and return predictions from the estimator.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        $this->preprocess($dataset);

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
        
        $this->preprocess($dataset);

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
     * Apply the transformer stack to a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function preprocess(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            $dataset->apply($transformer);
        }
    }

    /**
     * Allow methods to be called on the estimator from the wrapper.
     *
     * @param string $name
     * @param mixed[] $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        foreach ($arguments as $argument) {
            if ($argument instanceof Dataset) {
                $this->preprocess($argument);
            }
        }

        return $this->estimator->$name(...$arguments);
    }
}
