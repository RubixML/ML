<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Multiclass;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use PHPUnit\Framework\TestCase;

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
            costFn: new CrossEntropy()
        );

        $this->network = new Network(
            input: $this->input,
            hidden: $this->hidden,
            output: $this->output,
            optimizer: new Adam(0.001)
        );
    }

    public function testLayers() : void
    {
        $count = 0;

        foreach ($this->network->layers() as $item) {
            ++$count;
        }

        $this->assertSame(7, $count);
    }

    public function testInput() : void
    {
        $this->assertInstanceOf(Placeholder1D::class, $this->network->input());
    }

    public function testHidden() : void
    {
        $this->assertCount(5, $this->network->hidden());
    }

    public function testNumParams() : void
    {
        $this->network->initialize();

        $this->assertEquals(103, $this->network->numParams());
    }
}
