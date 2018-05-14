<?php

namespace Rubix\Engine\Estimators\Predictions;

class Probabalistic extends Prediction
{
    /**
     * The probability estimate of the prediction.
     *
     * @var float
     */
    protected $probability;

    /**
     * @param  mixed  $outcome
     * @param  float  $probability
     * @return void
     */
    public function __construct($outcome, float $probability)
    {
        $this->probability = $probability;

        parent::__construct($outcome);
    }

    /**
     * @return float
     */
    public function probability() : float
    {
        return $this->probability;
    }
}
