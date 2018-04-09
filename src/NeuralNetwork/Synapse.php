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
    protected $weight = 0.0;

    /**
     * Initialize the synapse with a random weight.
     *
     * @param  \Rubix\Engine\Neuron  $neuron
     * @return self
     */
    public static function init(Neuron $neuron, float $min = -4.0, float $max = 4.0) : self
    {
        $synapse = new static($neuron);

        $synapse->randomize($min, $max);

        return $synapse;
    }

    /**
     * @param  \Rubix\Engine\NeuralNetwork\Neuron  $neuron
     * @return void
     */
    public function __construct(Neuron $neuron)
    {
        $this->neuron = $neuron;
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
