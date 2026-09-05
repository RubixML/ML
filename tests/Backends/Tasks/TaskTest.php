<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Backends\Tasks;

use Amp\NullCancellation;
use Amp\Sync\Channel;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Backends\Tasks\Task;
use PHPUnit\Framework\TestCase;

#[Group('Backends')]
#[CoversClass(Task::class)]
class TaskTest extends TestCase
{
    #[Test]
    public function runDelegatesToCompute() : void
    {
        $task = new Task(
            fn: fn ($x) => $x * 2,
            args: [21]
        );

        $channel = $this->createStub(Channel::class);

        $cancellation = new NullCancellation();

        $this->assertEquals(42, $task->run($channel, $cancellation));
        $this->assertEquals($task->compute(), $task->run($channel, $cancellation));
    }
}
