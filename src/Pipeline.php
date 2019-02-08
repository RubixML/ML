<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Transformers\Elastic;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Other\Traits\LoggerAware;
use InvalidArgumentException;
use RuntimeException;

/**
 * Pipeline
 *
 * Pipeline is a meta estimator responsible for transforming the input
 * data by applying a series of transformer middleware. Pipeline accepts
 * a base estimator and a list of transformers to apply to the input
 * data before it is fed to the estimator. Under the hood, Pipeline will
 * automatically fit the training set upon training and transform any
 * Dataset object supplied as an argument to one of the base Estimator's
 * methods, including `train()` and `predict()`. With the *elastic* mode
 * enabled, Pipeline can update the fitting of certain transformers during
 * online (*partial*) training.
 *
 * > **Note**: Since transformations are applied to dataset objects in place
 * (without making a copy), using the dataset in a program after it has been
 * run through Pipeline may have unexpected results. If you need a *clean*
 * dataset object to call multiple methods with, you can use the PHP clone
 * syntax to keep an original (untransformed) copy in memory.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Pipeline implements Online, Wrapper, Probabilistic, Persistable, Verbose
{
    use LoggerAware;

    /**
     * The transformer middleware that preprocesses the data for the estimator.
     *
     * @var array
     */
    protected $transformers = [
        //
    ];

    /**
     * The wrapped estimator instance.
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
     * Whether or not the transformers have been fitted.
     *
     * @var bool
     */
    protected $fitted;

    /**
     * @param  array  $transformers
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  bool  $elastic
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $transformers, Estimator $estimator, bool $elastic = true)
    {
        foreach ($transformers as $transformer) {
            if (!$transformer instanceof Transformer) {
                throw new InvalidArgumentException('Pipeline only accepts'
                    . ' transformer middleware, ' . gettype($transformer)
                    . ' found.');
            }
        }

        $this->transformers = $transformers;
        $this->estimator = $estimator;
        $this->elastic = $elastic;
        $this->fitted = false;
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return int
     */
    public function type() : int
    {
        return $this->estimator->type();
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return $this->estimator->compatibility();
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return ($this->estimator instanceof Learner
            and $this->estimator->trained() and $this->fitted)
            or $this->fitted;
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
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        $this->fit($dataset);

        if ($this->estimator instanceof Learner) {
            $this->estimator->train($dataset);
        }

        $this->fitted = true;
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
     * Estimate probabilities for each possible outcome.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $base = $this->base();

        if ($base instanceof Probabilistic) {
            $this->preprocess($dataset);
            
            return $base->proba($dataset);
        }

        throw new RuntimeException('Base estimator must'
            . ' implement the probabilistic interface.');
    }

    /**
     * Fit the transformer middelware to a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    protected function fit(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Stateful) {
                if ($this->logger) {
                    $this->logger->info('Fitting '
                    . Params::shortName($transformer));
                }

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
    protected function update(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Elastic) {
                if ($this->logger) {
                    $this->logger->info('Updating '
                    . Params::shortName($transformer));
                }

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
    protected function preprocess(Dataset $dataset) : void
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
