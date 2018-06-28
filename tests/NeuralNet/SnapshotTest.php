<?php

namespace Rubix\Tests\NeuralNet;

use Rubix\ML\NeuralNet\Snapshot;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use PHPUnit\Framework\TestCase;

class SnapshotTest extends TestCase
{
    protected $snapshot;

    protected $layer;

    public function setUp()
    {
        $this->layer = new Dense(5, new Sigmoid());

        $this->layer->initialize(5);

        $this->snapshot = new Snapshot([$this->layer]);
    }

    public function test_build_snapshot()
    {
        $this->assertInstanceOf(Snapshot::class, $this->snapshot);
    }

    public function test_save_snapshot()
    {
        $this->assertFalse(file_exists(__DIR__ . '/test.snapshot'));

        $this->snapshot->save(__DIR__ . '/test.snapshot');

        $this->assertFileExists(__DIR__ . '/test.snapshot');
    }

    public function test_restore_snapshot()
    {
        $snapshot = Snapshot::restore(__DIR__ . '/test.snapshot');

        $this->assertInstanceOf(Snapshot::class, $snapshot);

        $this->assertTrue(unlink(__DIR__ . '/test.snapshot'));
    }
}
