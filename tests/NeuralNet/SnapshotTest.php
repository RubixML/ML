<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Snapshot;

#[Group('NeuralNet')]
#[CoversClass(Snapshot::class)]
class SnapshotTest extends TestCase
{
    protected Snapshot $snapshot;

    protected Network $network;

    public function testConstructorThrowsWithWrongParameters() : void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Number of layers and parameter groups must be equal.');

        new Snapshot(
            layers: [new Dense(1)],
            parameters: []
        );
    }

    public function testTake() : void
    {
        $network = new FeedForward(
            input: new Placeholder1D(1),
            hidden: [
                new Dense(10),
                new Activation(new ELU()),
                new Dense(5),
                new Activation(new ELU()),
                new Dense(1),
            ],
            output: new Binary(
                classes: ['yes', 'no'],
                costFn:  new BinaryCrossEntropy()
            ),
            optimizer: new Stochastic()
        );

        $network->initialize();

        $this->expectNotToPerformAssertions();

        Snapshot::take($network);
    }
}
