<?php

namespace Rubix\ML\Tests\Backends;

use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Backend;
use PHPUnit\Framework\TestCase;

class AmpTest extends TestCase
{
    protected $backend;

    public function setUp()
    {
        $this->backend = new Amp(4);
    }

    public function test_build_backend()
    {
        $this->assertInstanceOf(Amp::class, $this->backend);
        $this->assertInstanceOf(Backend::class, $this->backend);
    }

    public function test_autotune()
    {
        $backend = Amp::autotune();

        $this->assertInstanceOf(Amp::class, $backend);
        $this->assertInstanceOf(Backend::class, $backend);
    }

    public function test_enqueue_process()
    {
        $functions = array_fill(0, 10, [self::class, 'foo']);

        foreach ($functions as $i => $function) {
            $this->backend->enqueue($function, [$i]);
        }

        $results = $this->backend->process();

        $this->assertCount(10, $results);
    }

    public static function foo(int $i)
    {
        return [$i, microtime(true)];
    }
}
