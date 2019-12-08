<?php

namespace Rubix\ML\Tests\Backends;

use Rubix\ML\Deferred;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Backend;
use PHPUnit\Framework\TestCase;

class AmpTest extends TestCase
{
    /**
     * @var \Rubix\ML\Backends\Amp
     */
    protected $backend;

    public function setUp() : void
    {
        $this->backend = new Amp(4);
    }

    public function test_build_backend() : void
    {
        $this->assertInstanceOf(Amp::class, $this->backend);
        $this->assertInstanceOf(Backend::class, $this->backend);
    }

    public function test_workers() : void
    {
        $this->assertEquals(4, $this->backend->workers());
    }

    public function test_enqueue_process() : void
    {
        $functions = array_fill(0, 10, [self::class, 'foo']);

        foreach ($functions as $i => $function) {
            $this->backend->enqueue(new Deferred($function, [$i]));
        }

        $results = $this->backend->process();

        $this->assertCount(10, $results);
    }

    public static function foo(int $i) : array
    {
        return [$i, microtime(true)];
    }
}
