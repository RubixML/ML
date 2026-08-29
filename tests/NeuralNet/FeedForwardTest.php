<?php

namespace Rubix\ML\Tests\NeuralNet;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Optimizers\Adam;

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
     * @var Hidden[]
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
    #[TestDox('Builds a feed-forward network instance')]
    public function build() : void
    {
        self::assertInstanceOf(FeedForward::class, $this->network);
    }

    #[Test]
    #[TestDox('Returns all hidden and output layers')]
    public function layers() : void
    {
        self::assertCount(5, iterator_to_array($this->network->layers()));
    }

    #[Test]
    #[TestDox('Returns the input layer')]
    public function input() : void
    {
        self::assertInstanceOf(Placeholder1D::class, $this->network->input());
    }

    #[Test]
    #[TestDox('Returns the hidden layers')]
    public function hidden() : void
    {
        self::assertCount(5, $this->network->hidden());
    }

    #[Test]
    #[TestDox('Returns the output layer')]
    public function networkOutput() : void
    {
        self::assertInstanceOf(Output::class, $this->network->output());
    }

    #[Test]
    #[TestDox('Reports the correct number of parameters after initialization')]
    public function numParams() : void
    {
        $this->network->initialize();

        self::assertEquals(103, $this->network->numParams());
    }

    #[Test]
    #[TestDox('Performs a roundtrip pass and returns a loss value')]
    public function roundtrip() : void
    {
        $this->network->initialize();

        $loss = $this->network->roundtrip($this->dataset);

        self::assertIsFloat($loss);
    }
}
