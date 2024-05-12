<?php

namespace Rubix\ML\Tests\Backends;

use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Amp2;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Tasks\Task;
use PHPUnit\Framework\TestCase;

/**
 * @group Backends
 * @covers \Rubix\ML\Backends\Amp
 */
abstract class AbstractAmp extends TestCase
{
    /**
     * @var Amp|Amp2
     */
    protected $backend;

    /**
     * @param int $i
     * @return array<int|float>
     */
    public static function foo(int $i) : array
    {
        return [$i * 2, microtime(true)];
    }

    /**
     * @test
     */
    public function workers() : void
    {
        $this->assertEquals(4, $this->backend->workers());
    }

    /**
     * @test
     */
    public function enqueueProcess() : void
    {
        for ($i = 0; $i < 10; ++$i) {
            $this->backend->enqueue(new Task([self::class, 'foo'], [$i]));
        }

        $results = $this->backend->process();

        $this->assertCount(10, $results);
        array_map([$this, 'assertIsArray'], $results);
    }
}
