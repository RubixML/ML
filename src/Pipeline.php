<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Transformers\Elastic;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Other\Traits\LoggerAware;
use InvalidArgumentException;

/**
 * Pipeline
 *
 * Pipeline is responsible for transforming the input sample matrix of a Dataset
 * in such a way that can be processed by the base Estimator. Pipeline accepts a
 * base Estimator and a list of Transformers to apply to the input data before
 * it is fed to the learning algorithm. Under the hood, Pipeline will
 * automatically fit the training set upon training and transform any Dataset
 * object supplied as an argument to one of the base Estimator’s methods,
 * including predict().
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Pipeline implements MetaEstimator, Online, Verbose, Persistable
{
    use LoggerAware;

    /**
     * The wrapped estimator instance.
     *
     * @var \Rubix\ML\Estimator
     */
    protected $estimator;

    /**
     * The transformers that process the sample data before they are fed to the
     * estimator for training, testing, and prediction.
     *
     * @var array
     */
    protected $transformers = [
        //
    ];

    /**
     * Should we update elastic transformers during partial train?
     * 
     * @var bool
     */
    protected $elastic;

    /**
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  array  $transformers
     * @param  bool  $elastic
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(Estimator $estimator, array $transformers = [], bool $elastic = true)
    {
        foreach ($transformers as $transformer) {
            if (!$transformer instanceof Transformer) {
                throw new InvalidArgumentException('Pipeline only accepts'
                    . ' transformers as middleware, ' . gettype($transformer)
                    . ' found.');
            }

            $this->transformers[] = $transformer;
        }

        $this->estimator = $estimator;
        $this->elastic = $elastic;
    }

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->estimator->type();
    }

    /**
     * Return the base estimator instance.
     *
     * @return \Rubix\ML\Estimator
     */
    public function estimator() : Estimator
    {
        return $this->estimator;
    }

    /**
     * Run the training dataset through all transformers in order and use the
     * transformed dataset to train the estimator.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
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
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
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
     * Preprocess the dataset and return predictions from the estimator.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array
    {
        $this->preprocess($dataset);

        return $this->estimator->predict($dataset);
    }

    /**
     * Fit the transformer middelware to a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Stateful) {
                if ($this->logger) $this->logger->info('Fitting '
                    . Params::shortName($transformer));

                $transformer->fit($dataset);
            }

            $dataset->apply($transformer);
        }
    }

    /**
     * Update the fitting of the transformer middleware.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function update(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Elastic) {
                if ($this->logger) $this->logger->info('Updating '
                    . Params::shortName($transformer));

                $transformer->update($dataset);
            }

            $dataset->apply($transformer);
        }
    }

    /**
     * Apply the transformer middleware over a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
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
     * @param  string  $name
     * @param  array  $arguments
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
