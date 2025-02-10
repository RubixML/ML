<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Backends;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Backends\Swoole as SwooleBackend;
use Rubix\ML\Backends\Tasks\Task;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Specifications\SwooleExtensionIsLoaded;
use Swoole\Event;

#[Group('backends')]
#[Group('Swoole')]
#[CoversClass(SwooleBackend::class)]
class SwooleTest extends TestCase
{
    protected SwooleBackend $backend;

    public static function foo(int $i) : int
    {
        return $i * 2;
    }

    protected function setUp() : void
    {
        if (!SwooleExtensionIsLoaded::create()->passes()) {
            $this->markTestSkipped('Swoole extension is not available.');
        }

        $this->backend = new SwooleBackend();
    }

    protected function tearDown() : void
    {
        Event::wait();
    }

    public function testEnqueueProcess() : void
    {
        for ($i = 0; $i < 10; ++$i) {
            $this->backend->enqueue(
                task: new Task(
                    fn: [self::class, 'foo'],
                    args: [$i]
                )
            );
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
