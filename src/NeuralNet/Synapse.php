<?php

namespace Rubix\Engine\NeuralNet;

class Synapse
{
    /**
     * The node that this synapse connects to.
     *
     * @var  \Rubix\Engine\NeuralNet\Node
     */
    protected $node;

    /**
     * The weight of the connection.
     *
     * @var float
     */
    protected $weight;

    /**
     * @param  \Rubix\Engine\NeuralNet\Node  $node
     * @param  float  $weight
     * @return void
     */
    public function __construct(Node $node, float $weight = 0.0)
    {
        $this->node = $node;
        $this->weight = $weight;
    }

    /**
     * @return \Rubix\Engine\NeuralNetwork\Node
     */
    public function node() : Node
    {
        return $this->node;
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
        return $this->weight * $this->node->output();
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
        $scale = pow(10, 10);

        $this->weight = random_int($min * $scale, $max * $scale) / $scale;

        return $this;
    }
}
