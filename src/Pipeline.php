<?php

namespace Rubix\Engine;

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Estimator;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Estimators\Predictions\Prediction;
use RuntimeException;

class Pipeline implements Estimator, Persistable
{
    /**
     * The estimator.
     *
     * @var \Rubix\Engine\Estimators\Estimator
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
     * @param  \Rubix\Engine\Estimators\Estimator  $estimator
     * @param  array  $transformers
     * @return void
     */
    public function __construct(Estimator $estimator, array $transformers = [])
    {
        foreach ($transformers as $transformer) {
            $this->addTransformer($transformer);
        }

        $this->estimator = $estimator;
    }

    /**
     * Return the underlying estimator instance.
     *
     * @return \Rubix\Engine\Estimators\Estimator
     */
    public function estimator() : Estimator
    {
        return $this->estimator;
    }

    /**
     * Run the training dataset through all transformers in order and use the
     * transformed dataset to train the estimator.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \RuntimeException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            $transformer->fit($dataset);

            $dataset->transform($transformer);
        }

        $this->estimator->train($dataset);
    }

    /**
     * Preprocess the sample and make a prediction.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Estimaotors\Predictions\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $samples = [$sample];

        foreach ($this->transformers as $transformer) {
            $transformer->transform($samples);
        }

        return $this->estimator->predict($samples[0]);
    }

    /**
     * Add a transformer middleware to the pipeline.
     *
     * @param  \Rubix\Engine\Contracts\Transformer  $transformer
     * @return self
     */
    public function addTransformer(Transformer $transformer) : self
    {
        $this->transformers[] = $transformer;

        return $this;
    }

    /**
     * Allow methods to be called from the base estimator from this estimator.
     *
     * @param  string  $name
     * @param  array  $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        return $this->estimator->$name(...$arguments);
    }
}
