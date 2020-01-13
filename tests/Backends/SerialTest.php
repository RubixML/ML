<?php

namespace Rubix\ML\Tests\Backends;

use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Deferred;
use PHPUnit\Framework\TestCase;

/**
 * @group Backends
 * @covers \Rubix\ML\Backends\Serial
 */
class SerialTest extends TestCase
{
    /**
     * @var \Rubix\ML\Backends\Serial
     */
    protected $backend;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->backend = new Serial();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Serial::class, $this->backend);
        $this->assertInstanceOf(Backend::class, $this->backend);
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
