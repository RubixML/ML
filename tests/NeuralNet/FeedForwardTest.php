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
    /**
     * @var \Rubix\ML\Datasets\Labeled
     */
    protected $dataset;

    /**
     * @var \Rubix\ML\NeuralNet\FeedForward
     */
    protected $network;

    /**
     * @var \Rubix\ML\NeuralNet\Layers\Input
     */
    protected $input;

    /**
     * @var \Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    protected $hidden;

    /**
     * @var \Rubix\ML\NeuralNet\Layers\Output
     */
    protected $output;

    public function setUp() : void
    {
        $this->dataset = new Labeled([
            [1., 2.5,],
            [0.1, 0.],
            [0.002, -6.],
        ], ['yes', 'no', 'maybe'], false);

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

    public function test_build_network() : void
    {
        $this->assertInstanceOf(FeedForward::class, $this->network);
        $this->assertInstanceOf(Network::class, $this->network);
    }

    public function test_depth() : void
    {
        $this->assertEquals(6, $this->network->depth());
    }

    public function test_get_input_layer() : void
    {
        $this->assertInstanceOf(Placeholder1D::class, $this->network->input());
    }

    public function test_get_hidden_layers() : void
    {
        $this->assertCount(4, $this->network->hidden());
    }

    public function test_get_output_layer() : void
    {
        $this->assertInstanceOf(Output::class, $this->network->output());
    }

    public function test_get_parametric_layers() : void
    {
        $this->assertCount(3, $this->network->parametric());
    }

    public function test_roundtrip() : void
    {
        $loss = $this->network->roundtrip($this->dataset);

        $this->assertIsFloat($loss);
    }
}
