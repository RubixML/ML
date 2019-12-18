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

class SnapshotTest extends TestCase
{
    /**
     * @var \Rubix\ML\NeuralNet\Snapshot<array>
     */
    protected $snapshot;

    /**
     * @var \Rubix\ML\NeuralNet\Network
     */
    protected $network;

    public function setUp() : void
    {
        $this->network = new FeedForward(new Placeholder1D(1), [
            new Dense(10),
            new Activation(new ELU()),
        ], new Binary(['yes', 'no'], 1e-4, new CrossEntropy()), new Stochastic());

        $this->snapshot = new Snapshot($this->network);
    }

    public function test_build_snapshot() : void
    {
        $this->assertInstanceOf(Snapshot::class, $this->snapshot);
        $this->assertCount(2, iterator_to_array($this->snapshot));
        $this->assertEquals(4, $this->network->depth());
    }
}
