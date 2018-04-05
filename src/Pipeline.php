<?php

namespace Rubix\Engine;

use Rubix\Engine\Preprocessors\Preprocessor;

class Pipeline implements Estimator
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
     * Train the pipeline.
     *
     * @param  \Rubix\Engine\SupervisedDataset  $data
     * @return void
     */
    public function train(SupervisedDataset $data) : void
    {
        list($samples, $outcomes) = $data->toArray();

        foreach ($this->preprocessors as $preprocessor) {
            $preprocessor->fit($samples, $outcomes);
            $preprocessor->transform($samples);
        }

        $this->estimator->train(new SupervisedDataset($samples, $outcomes));
    }

    /**
     * Make a prediction of a given sample.
     *
     * @param  array  $sample
     * @return array
     */
    public function predict(array $sample) : array
    {
        $sample = [$sample];

        foreach ($this->preprocessors as $preprocessor) {
            $preprocessor->transform($sample);
        }

        return $this->estimator->predict($sample);
    }

    /**
     * Add a preprocessor middleware to the pipline.
     *
     * @param  \Rubix\Engine\Preprocessors\Preprocessor  $preprocessor
     * @return self
     */
    public function addPreprocessor(Preprocessor $preprocessor) : self
    {
        $this->preprocessors[] = $preprocessor;

        return $this;
    }
}
