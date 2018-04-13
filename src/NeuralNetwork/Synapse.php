<?php

namespace Rubix\Engine\NeuralNetwork;

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
     * @param  float  $weight
     * @return void
     */
    public function __construct(Neuron $neuron, float $weight = 0.0)
    {
        $this->neuron = $neuron;
        $this->weight = $weight;
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
     * Set the weight to the given value.
     *
     * @param  float  $weight
     * @return self
     */
    public function setWeight(float $weight) : self
    {
        $this->weight = $weight;

        return $this;
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
     * Randomize the weight of the synapse.
     *
     * @param  float  $min
     * @param  float  $max
     * @return self
     */
    public function randomize(float $min = -4.0, float $max = 4.0) : self
    {
        $scale = pow(10, 8);

        $this->weight = random_int($min * $scale, $max * $scale) / $scale;

        return $this;
    }
}
