<?php

namespace Rubix\Engine;

use Rubix\Engine\Preprocessors\Preprocessor;
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
    protected $preprocessors = [
        //
    ];

    /**
     * @param  \Rubix\Engine\Estimator  $estimator
     * @param  array  $preprocessors
     * @return void
     */
    public function __construct(Estimator $estimator, array $preprocessors = [])
    {
        foreach ($preprocessors as $preprocessor) {
            $this->addPreprocessor($preprocessor);
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
     * Run the training dataset through all preprocessors in order and use the
     * transformed dataset to train the estimator.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @throws \RuntimeException
     * @return void
     */
    public function train(Dataset $data) : void
    {
        foreach ($this->preprocessors as $preprocessor) {
            $preprocessor->fit($data);

            $data->transform($preprocessor);
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

        foreach ($this->preprocessors as $preprocessor) {
            $preprocessor->transform($samples);
        }

        return $this->estimator->predict($samples[0]);
    }

    /**
     * Add a preprocessor middleware to the pipeline.
     *
     * @param  \Rubix\Engine\Contracts\Preprocessor  $preprocessor
     * @return self
     */
    public function addPreprocessor(Preprocessor $preprocessor) : self
    {
        $this->preprocessors[] = $preprocessor;

        return $this;
    }
}
