<?php

namespace Rubix\Engine\NeuralNetwork;

use Rubix\Engine\NeuralNetwork\ActivationFunctions\Sigmoid;
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
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct($outcome)
    {
        if (!is_numeric($outcome) && !is_string($outcome)) {
            throw new InvalidArgumentException('Outcome must be a numeric or string value, ' . gettype($outcome) . ' found.');
        }

        $this->outcome = $outcome;

        parent::__construct(new Sigmoid());
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
}
