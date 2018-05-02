<?php

namespace Rubix\Engine;

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Transformers\Strategies\Strategy;
use Rubix\Engine\Transformers\Strategies\Continuous;
use Rubix\Engine\Transformers\Strategies\Categorical;

class DummyEstimator implements Estimator, Classifier, Regression
{
    /**
     * The guessing strategy the estimator employs.
     *
     * @var \Rubix\Engine\Transformers\Strategies\Strategy
     */
    protected $strategy;

    /**
     * @param  \Rubix\Engine\Transformers\Strategies\Strategy  $strategy
     * @return void
     */
    public function __construct(Strategy $strategy)
    {
        $this->strategy = $strategy;
    }

    /**
     * Train the estimator.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @throws \RuntimeException
     * @return void
     */
    public function train(Supervised $dataset) : void
    {
        if ($this->strategy instanceof Continuous) {
            if ($dataset->outcomeType() !== self::CONTINUOUS) {
                throw new RuntimeException('This estimator required continuous outcomes.');
            }
        } else if ($this->strategy instanceof Categorical) {
            if ($dataset->outcomeType() !== self::CATEGORICAL) {
                throw new RuntimeException('This estimator required categorical outcomes.');
            }
        }

        $this->strategy->fit($dataset->outcomes());
    }

    /**
     * Make a prediction of a given sample.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        return new Prediction($this->strategy->guess());
    }
}
