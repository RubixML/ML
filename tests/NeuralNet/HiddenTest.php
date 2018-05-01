<?php

use Rubix\Engine\NeuralNet\Node;
use Rubix\Engine\NeuralNet\Neuron;
use Rubix\Engine\NeuralNet\ActivationFunctions\Sigmoid;
use PHPUnit\Framework\TestCase;

class NeuronTest extends TestCase
{
    protected $neuron;

    public function setUp()
    {
        $this->neuron = new Neuron(new Sigmoid());
    }

    public function test_build_hidden_neuron()
    {
        $this->assertInstanceOf(Neuron::class, $this->neuron);
        $this->assertInstanceOf(Node::class, $this->neuron);
    }
}
