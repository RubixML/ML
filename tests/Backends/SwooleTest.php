<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Backends;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use PHPUnit\Framework\Attributes\RunInSeparateProcess;
use Rubix\ML\Backends\Swoole as SwooleBackend;
use Rubix\ML\Backends\Tasks\Task;
use PHPUnit\Framework\TestCase;
use Swoole\Event;

#[Group('backends')]
#[Group('Swoole')]
#[RequiresPhpExtension('swoole')]
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
        $this->backend = new SwooleBackend(4);
    }

    protected function tearDown() : void
    {
        Event::wait();
    }

    #[Test]
    #[RunInSeparateProcess]
    public function workers() : void
    {
        $this->assertEquals(4, $this->backend->workers());
    }

    #[Test]
    #[RunInSeparateProcess]
    public function enqueueProcess() : void
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
