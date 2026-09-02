<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
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
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Optimizers\Adam;

#[Group('NeuralNet')]
#[CoversClass(Network::class)]
class NetworkTest extends TestCase
{
    protected Labeled $dataset;

    protected Network $network;

    protected Input $input;

    /**
     * @var Hidden[]
     */
    protected array $hidden;

    protected Output $output;

    protected function setUp() : void
    {
        $this->dataset = Labeled::quick(
            samples: [
                [1.0, 2.5],
                [0.1, 0.0],
                [0.002, -6.0],
            ],
            labels: ['yes', 'no', 'maybe']
        );

        $this->input = new Placeholder1D(2);

        $this->hidden = [
            new Dense(neurons: 10),
            new Activation(new ReLU()),
            new Dense(neurons: 5),
            new Activation(new ReLU()),
            new Dense(neurons: 3),
        ];

        $this->output = new Multiclass(
            classes: ['yes', 'no', 'maybe'],
            costFn: new MulticlassCrossEntropy()
        );

        $this->network = new FeedForward(
            input: $this->input,
            hidden: $this->hidden,
            output: $this->output,
            optimizer: new Adam(0.001),
            dataType: 'float32'
        );
    }

    #[Test]
    public function layers() : void
    {
        $count = 0;

        foreach ($this->network->layers() as $item) {
            ++$count;
        }

        self::assertSame(7, $count);
    }

    #[Test]
    public function input() : void
    {
        self::assertInstanceOf(Placeholder1D::class, $this->network->input());
    }

    #[Test]
    public function hidden() : void
    {
        self::assertCount(5, $this->network->hidden());
    }

    #[Test]
    public function numParams() : void
    {
        $this->network->initialize();

        self::assertEquals(103, $this->network->numParams());
    }
}
