<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet\Snapshots;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU\ELU;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy\CrossEntropy;
use Rubix\ML\NeuralNet\Layers\Activation\Activation;
use Rubix\ML\NeuralNet\Layers\Binary\Binary;
use Rubix\ML\NeuralNet\Layers\Dense\Dense;
use Rubix\ML\NeuralNet\Layers\Placeholder1D\Placeholder1D;
use Rubix\ML\NeuralNet\Networks\Base\Contracts\Network;
use Rubix\ML\NeuralNet\Networks\FeedForward\FeedForward;
use Rubix\ML\NeuralNet\Optimizers\Stochastic\Stochastic;
use Rubix\ML\NeuralNet\Snapshots\Snapshot;

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
                costFn:  new CrossEntropy()
            ),
            optimizer: new Stochastic()
        );

        $network->initialize();

        $this->expectNotToPerformAssertions();

        Snapshot::take($network);
    }
}
