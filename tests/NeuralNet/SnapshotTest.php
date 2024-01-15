<?php

namespace Rubix\ML\Tests\NeuralNet;

use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use PHPUnit\Framework\TestCase;

/**
 * @group NeuralNet
 * @covers \Rubix\ML\NeuralNet\Snapshot
 */
class SnapshotTest extends TestCase
{
    /**
     * @var Snapshot
     */
    protected $snapshot;

    /**
     * @var \Rubix\ML\NeuralNet\Network
     */
    protected $network;

    /**
     * @test
     */
    public function take() : void
    {
        $network = new FeedForward(new Placeholder1D(1), [
            new Dense(10),
            new Activation(new ELU()),
            new Dense(5),
            new Activation(new ELU()),
            new Dense(1),
        ], new Binary(['yes', 'no'], new CrossEntropy()), new Stochastic());

        $network->initialize();

        $snapshot = Snapshot::take($network);

        $this->assertInstanceOf(Snapshot::class, $snapshot);
    }
}
