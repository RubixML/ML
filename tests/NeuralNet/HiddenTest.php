<?php

use Rubix\Engine\NeuralNet\Hidden;
use Rubix\Engine\NeuralNet\ActivationFunctions\Sigmoid;
use PHPUnit\Framework\TestCase;

class HiddenTest extends TestCase
{
    protected $neuron;

    public function setUp()
    {
        $this->neuron = new Hidden(new Sigmoid());
    }

    public function test_build_hidden_neuron()
    {
        $this->assertInstanceOf(Hidden::class, $this->neuron);
    }
}
