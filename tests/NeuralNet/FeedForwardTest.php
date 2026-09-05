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
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Layers\Input;

#[Group('NeuralNet')]
#[CoversClass(FeedForward::class)]
class FeedForwardTest extends TestCase
{
    /**
     * @var Labeled
     */
    protected Labeled $dataset;

    /**
     * @var FeedForward
     */
    protected FeedForward $network;

    /**
     * @var Input
     */
    protected Input $input;

    /**
     * @var \Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    protected array $hidden;

    /**
     * @var Output
     */
    protected Output $output;

    protected function setUp() : void
    {
        $this->dataset = Labeled::quick([
            [1.0, 2.5],
            [0.1, 0.0],
            [0.002, -6.0],
        ], ['yes', 'no', 'maybe']);

        $this->input = new Placeholder1D(2);

        $this->hidden = [
            new Dense(10),
            new Activation(new ReLU()),
            new Dense(5),
            new Activation(new ReLU()),
            new Dense(3),
        ];

        $this->output = new Multiclass(['yes', 'no', 'maybe'], new MulticlassCrossEntropy());

        $this->network = new FeedForward($this->input, $this->hidden, $this->output, new Adam(0.001));
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(FeedForward::class, $this->network);
        $this->assertInstanceOf(Network::class, $this->network);
    }

    #[Test]
    public function layers() : void
    {
        $this->assertCount(5, iterator_to_array($this->network->layers()));
    }

    #[Test]
    public function input() : void
    {
        $this->assertInstanceOf(Placeholder1D::class, $this->network->input());
    }

    #[Test]
    public function hidden() : void
    {
        $this->assertCount(5, $this->network->hidden());
    }

    #[Test]
    public function testOutput() : void
    {
        $this->assertInstanceOf(Output::class, $this->network->output());
    }

    #[Test]
    public function numParams() : void
    {
        $this->network->initialize();

        $this->assertEquals(103, $this->network->numParams());
    }

    #[Test]
    public function roundtrip() : void
    {
        $this->network->initialize();

        $loss = $this->network->roundtrip($this->dataset);

        $this->assertIsFloat($loss);
    }
}
