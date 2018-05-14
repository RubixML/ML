<?php

namespace Rubix\Engine\Estimators\Predictions;

class Prediction
{
    /**
     * The outcome of a prediction.
     *
     * @var mixed
     */
    protected $outcome;

    /**
     * @param  mixed  $outcome
     * @return void
     */
    public function __construct($outcome)
    {
        $this->outcome = $outcome;
    }

    /**
     * @return mixed
     */
    public function outcome()
    {
        return $this->outcome;
    }

    /**
     * The output of the prediction is categorical.
     *
     * @return bool
     */
    public function categorical() : bool
    {
        return is_string($this->outcome);
    }

    /**
     * The output of the prediction is continuous.
     *
     * @return bool
     */
    public function continuous() : bool
    {
        return !$this->categorical() && is_numeric($this->outcome);
    }
}
