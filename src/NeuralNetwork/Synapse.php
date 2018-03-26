<?php

namespace Rubix\Engine\NeuralNetwork;

use Rubix\Engine\Math\Random;

class Synapse
{
    /**
     * The neuron that this synapse connects to.
     *
     * @var  \Rubix\Engine\NeuralNetwork\Neuron
     */
    protected $neuron;

    /**
     * The weight of the connection.
     *
     * @var float
     */
    protected $weight;

    /**
     * @param  \Rubix\Engine\NeuralNetwork\Neuron  $neuron
     * @return void
     */
    public function __construct(Neuron $neuron)
    {
        $this->neuron = $neuron;

        $this->randomize();
    }

    /**
     * @return \Rubix\Engine\NeuralNetowkr\Neuron
     */
    public function neuron() : Neuron
    {
        return $this->neuron;
    }

    /**
     * @return float
     */
    public function weight() : float
    {
        return $this->weight;
    }

    /**
     * The value of the impulse being sent from the connected neuron.
     *
     * @return float
     */
    public function impulse() : float
    {
        return $this->weight * $this->neuron->output();
    }

    /**
     * Adjust the weight by delta. i.e. the difference.
     *
     * @param  float  $delta
     * @return self
     */
    public function adjustWeight(float $delta) : self
    {
        $this->weight += $delta;

        return $this;
    }

    /**
     * Randomize the weight.
     *
     * @return self
     */
    public function randomize() : self
    {
        $this->weight = Random::float(-4, 4, 5);

        return $this;
    }
}
