<?php

namespace Rubix\ML\Tests\Backends;

use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Tasks\Task;
use PHPUnit\Framework\TestCase;

/**
 * @group Backends
 * @covers \Rubix\ML\Backends\Amp
 */
class AmpTest extends TestCase
{
    /**
     * @var Amp
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
     * @before
     */
    protected function setUp() : void
    {
        $this->backend = new Amp(4);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Amp::class, $this->backend);
        $this->assertInstanceOf(Backend::class, $this->backend);
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
    }
}
