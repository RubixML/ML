<?php

namespace Rubix\Engine;

use Rubix\Engine\Transformers\Transformer;
use Rubix\Engine\Connectors\Persistable;
use RuntimeException;

class Pipeline implements Estimator, Persistable
{
    /**
     * The estimator.
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
     * @param  \Rubix\Engine\Dataset  $data
     * @throws \RuntimeException
     * @return void
     */
    public function train(Dataset $data) : void
    {
        foreach ($this->transformers as $transformer) {
            $transformer->fit($data);

            $data->transform($transformer);
        }

        $this->estimator->train($data);
    }

    /**
     * Preprocess the sample and make a prediction.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
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
}
