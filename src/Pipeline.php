<?php

namespace Rubix\Engine;

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Clusterers\Clusterer;
use Rubix\Engine\Regressors\Regressor;
use Rubix\Engine\Classifiers\Classifier;
use Rubix\Engine\Transformers\Transformer;

class Pipeline implements Classifier, Clusterer, Regressor, Persistable
{
    /**
     * The wrapped estimator instance.
     *
     * @var \Rubix\Engine\Estimator
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
     * @param  \Rubix\Engine\Estimator  $estimator
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
     * Return the underlying model instance.
     *
     * @return \Rubix\Engine\Estimator
     */
    public function estimator() : Estimator
    {
        return $this->estimator;
    }

    /**
     * Run the training dataset through all transformers in order and use the
     * transformed dataset to train the estimator.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            $transformer->fit($dataset);

            $dataset->transform($transformer);
        }

        $this->estimator->train($dataset);
    }

    /**
     * Preoprecess the sample dataset and make a prediction.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $this->preprocess($samples);

        return $this->estimator->predict($samples);
    }

    /**
     * Add a transformer middleware to the pipeline.
     *
     * @param  \Rubix\Engine\Transformers\Transformer  $transformer
     * @return self
     */
    public function addTransformer(Transformer $transformer) : self
    {
        $this->transformers[] = $transformer;

        return $this;
    }

    /**
     * Run the transformer middleware over a dataset.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @return void
     */
    public function preprocess(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            $dataset->transform($transformer);
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
