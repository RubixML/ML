<?php

namespace Rubix\Engine\Estimators\Wrappers;

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Estimator;
use Rubix\Engine\Estimators\Persistable;
use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Estimators\Predictions\Prediction;
use RuntimeException;

class Pipeline extends Wrapper implements Estimator, Persistable
{
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

        parent::__construct($estimator);
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
     * Preoprecess the sample dataset and make a prediction.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        foreach ($this->transformers as $transformer) {
            $samples->transform($transformer);
        }

        return $this->estimator->predict($samples);
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
}
