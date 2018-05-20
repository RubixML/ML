<?php

namespace Rubix\Engine\Estimators;

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Persistable;
use Rubix\Engine\Estimators\Predictions\Prediction;
use Rubix\Engine\Transformers\Strategies\Continuous;

class DummyRegressor implements Regressor, Persistable
{
    /**
     * The guessing strategy that the dummy employs.
     *
     * @var \Rubix\Engine\Transformers\Strategies\Continuous
     */
    protected $strategy;

    /**
     * @param  \Rubix\Engine\Transformers\Strategies\Continuous  $strategy
     * @return void
     */
    public function __construct(Continuous $strategy)
    {
        $this->strategy = $strategy;
    }

    /**
     * Fit the training set to the given guessing strategy.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        $this->strategy->fit($dataset->labels());
    }

    /**
     * Make a prediction of a given sample dataset.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array
    {
        $predictions = [];

        foreach ($samples as $sample) {
            $predictions[] = new Prediction($this->strategy->guess());
        }

        return $predictions;
    }
}
