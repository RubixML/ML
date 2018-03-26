<?php

use Rubix\Engine\NeuralNetwork\Bias;
use PHPUnit\Framework\TestCase;

class BiasTest extends TestCase
{
    protected $neuron;

    public function setUp()
    {
        $this->neuron = new Bias();
    }

    public function test_build_bias_neuron()
    {
        $this->assertInstanceOf(Bias::class, $this->neuron);
    }

    public function test_output()
    {
        $this->assertEquals(1.0, $this->neuron->output());
    }
}
