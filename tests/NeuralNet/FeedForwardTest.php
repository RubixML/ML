<?php

namespace Rubix\ML\Tests\NeuralNet;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use PHPUnit\Framework\TestCase;

class FeedForwardTest extends TestCase
{
    protected $dataset;

    protected $activations;

    protected $network;

    protected $input;

    protected $hidden;

    protected $output;

    public function setUp()
    {
        $this->dataset = new Labeled([
            [1., 2.5,],
            [0.1, 0.],
            [0.002, -6.],
        ], ['yes', 'no', 'maybe'], false);

        $this->activations = [
            [1.0, 2.5],
            [0.1, 0.0],
            [0.002, 0.],
        ];

        $this->input = new Placeholder1D(2);

        $this->hidden = [
            new Dense(5),
            new Activation(new ReLU()),
            new Dense(5),
            new Activation(new ReLU()),
        ];

        $this->output = new Multiclass(['yes', 'no', 'maybe'], 1e-4, new CrossEntropy());

        $this->network = new FeedForward($this->input, $this->hidden, $this->output, new Adam(0.001));
    }

    public function test_build_network()
    {
        $this->assertInstanceOf(FeedForward::class, $this->network);
        $this->assertInstanceOf(Network::class, $this->network);
    }

    public function test_depth()
    {
        $this->assertEquals(6, $this->network->depth());
    }

    public function test_get_input_layer()
    {
        $this->assertInstanceOf(Placeholder1D::class, $this->network->input());
    }

    public function test_get_hidden_layers()
    {
        $this->assertCount(4, $this->network->hidden());
    }

    public function test_get_output_layer()
    {
        $this->assertInstanceOf(Output::class, $this->network->output());
    }

    public function test_get_parametric_layers()
    {
        $this->assertCount(3, $this->network->parametric());
    }

    public function test_round_trip()
    {
        $loss = $this->network->roundtrip($this->dataset);

        $this->assertEquals(0., $loss, '', INF);
    }
}
