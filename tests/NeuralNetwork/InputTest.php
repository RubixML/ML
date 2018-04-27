<?php

use Rubix\Engine\NeuralNet\Input;
use PHPUnit\Framework\TestCase;

class InputTest extends TestCase
{
    protected $neuron;

    public function setUp()
    {
        $this->neuron = new Input();
    }

    public function test_build_input_neuron()
    {
        $this->assertInstanceOf(Input::class, $this->neuron);
    }

    public function test_prime_input_neuron()
    {
        $this->assertEquals(0.0, $this->neuron->output());

        $this->neuron->prime(1.0);

        $this->assertEquals(1.0, $this->neuron->output());
    }
}
