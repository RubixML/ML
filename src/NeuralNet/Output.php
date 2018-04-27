<?php

namespace Rubix\Engine\NeuralNet;

use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use InvalidArgumentException;

class Output extends Hidden
{
    /**
     * The outcome that a particular output neuron measures.
     *
     * @var mixed
     */
    protected $outcome;

    /**
     * @param  mixed  $outcome
     * @param  \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction  $activationFunction
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($outcome, ActivationFunction $activationFunction)
    {
        if (!is_numeric($outcome) && !is_string($outcome)) {
            throw new InvalidArgumentException('Outcome must be a numeric or string value, ' . gettype($outcome) . ' found.');
        }

        $this->outcome = $outcome;

        parent::__construct($activationFunction);
    }

    /**
     * Function to kick off a recursive call to output() for each connected neuron.
     *
     * @return float
     */
    public function fire() : float
    {
        return $this->output();
    }

    /**
     * @return mixed
     */
    public function outcome()
    {
        return $this->outcome;
    }

    /**
     * Alias of output().
     *
     * @return float
     */
    public function activation() : float
    {
        return $this->output();
    }
}
