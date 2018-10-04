<?php

namespace Rubix\ML\Tests\NeuralNet;

use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Binary;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Placeholder;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use PHPUnit\Framework\TestCase;

class SnapshotTest extends TestCase
{
    protected $snapshot;

    protected $network;

    public function setUp()
    {
        $this->network = new FeedForward(new Placeholder(1), [
            new Dense(5),
            new Activation(new ELU()),
        ], new Binary(['yes', 'no']), new LeastSquares(), new Stochastic());

        $this->snapshot = Snapshot::take($this->network);
    }

    public function test_build_snapshot()
    {
        $this->assertInstanceOf(Snapshot::class, $this->snapshot);
    }
}
