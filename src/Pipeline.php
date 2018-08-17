<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Transformers\Transformer;

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
class Pipeline implements MetaEstimator, Persistable
{
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
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  array  $transformers
     * @return void
     */
    public function __construct(Estimator $estimator, array $transformers)
    {
        foreach ($transformers as $transformer) {
            $this->addTransformer($transformer);
        }

        $this->estimator = $estimator;
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
     * Return the underlying model instance.
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
        foreach ($this->transformers as $transformer) {
            $transformer->fit($dataset);

            $dataset->apply($transformer);
        }

        $this->estimator->train($dataset);
    }

    /**
     * Preoprecess the sample dataset and make a prediction.
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
     * Run the transformer middleware over a dataset.
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
     * Add a transformer middleware to the pipeline.
     *
     * @param  \Rubix\ML\Transformers\Transformer  $transformer
     * @return void
     */
    protected function addTransformer(Transformer $transformer) : void
    {
        $this->transformers[] = $transformer;
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
