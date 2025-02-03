<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\NeuralNet;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use PHPUnit\Framework\TestCase;

#[Group('NeuralNet')]
#[CoversClass(Snapshot::class)]
class SnapshotTest extends TestCase
{
    protected Snapshot $snapshot;

    protected Network $network;

    public function testTake() : void
    {
        $network = new Network(
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
