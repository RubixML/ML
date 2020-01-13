<?php

namespace Rubix\ML\Tests\Backends;

use Rubix\ML\Deferred;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Backend;
use PHPUnit\Framework\TestCase;

/**
 * @group Backends
 * @covers \Rubix\ML\Backends\Amp
 */
class AmpTest extends TestCase
{
    /**
     * @var \Rubix\ML\Backends\Amp
     */
    protected $backend;

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
        $functions = array_fill(0, 10, [self::class, 'foo']);

        foreach ($functions as $i => $function) {
            $this->backend->enqueue(new Deferred($function, [$i]));
        }

        $results = $this->backend->process();

        $this->assertCount(10, $results);
    }

    /**
     * @param int $i
     * @return array<int|float>
     */
    public static function foo(int $i) : array
    {
        return [$i, microtime(true)];
    }
}
