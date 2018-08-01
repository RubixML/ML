<?php

namespace Rubix\Tests\NeuralNet;

use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use PHPUnit\Framework\TestCase;

class SnapshotTest extends TestCase
{
    protected $snapshot;

    protected $layer;

    public function setUp()
    {
        $this->layer = new Dense(5, new ELU());

        $this->layer->init(5);

        $this->snapshot = new Snapshot([$this->layer]);
    }

    public function test_build_snapshot()
    {
        $this->assertInstanceOf(Snapshot::class, $this->snapshot);
    }
}
