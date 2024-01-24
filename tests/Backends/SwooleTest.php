<?php

namespace Rubix\ML\Tests\Backends;

use Rubix\ML\Backends\Swoole as SwooleBackend;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Tasks\Task;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Specifications\SwooleExtensionIsLoaded;
use Swoole\Event;

/**
 * @group Backends
 * @group Swoole
 * @covers \Rubix\ML\Backends\Swoole
 */
class SwooleTest extends TestCase
{
    /**
     * @var SwooleBackend
     */
    protected $backend;

    /**
     * @param int $i
     * @return int
     */
    public static function foo(int $i) : int
    {
        return $i * 2;
    }

    /**
     * @before
     */
    protected function setUp() : void
    {
        if (!SwooleExtensionIsLoaded::create()->passes()) {
            $this->markTestSkipped(
                'Swoole/OpenSwoole extension is not available.'
            );
        }

        $this->backend = new SwooleBackend();
    }

    /**
     * @after
     */
    protected function tearDown() : void
    {
        Event::wait();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(SwooleBackend::class, $this->backend);
        $this->assertInstanceOf(Backend::class, $this->backend);
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
        $this->assertEquals([
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
        ], $results);
    }
}
