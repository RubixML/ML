<?php

namespace Rubix\Tests\NeuralNet;

use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Layers\Placeholder;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use PHPUnit\Framework\TestCase;

class FeedForwardTest extends TestCase
{
    protected $network;

    protected $input;

    protected $hidden;

    protected $output;

    public function setUp()
    {
        $this->input = new Placeholder(5);

        $this->hidden = [
            new Dense(5, new ELU()),
            new Dense(5, new ELU()),
        ];

        $this->output = new Multiclass(['yes', 'no', 'maybe']);

        $this->network = new FeedForward($this->input, $this->hidden, $this->output, new Adam(0.001));
    }

    public function test_build_network()
    {
        $this->assertInstanceOf(FeedForward::class, $this->network);
        $this->assertInstanceOf(Network::class, $this->network);
    }

    public function test_depth()
    {
        $this->assertEquals(4, $this->network->depth());
    }

    public function test_get_input_layer()
    {
        $this->assertInstanceOf(Placeholder::class, $this->network->input());
    }

    public function test_get_hidden_layers()
    {
        $this->assertCount(2, $this->network->hidden());
    }

    public function test_get_output_layer()
    {
        $this->assertInstanceOf(Output::class, $this->network->output());
    }

    public function test_get_parametric_layers()
    {
        $this->assertCount(3, $this->network->parametric());
    }
}
