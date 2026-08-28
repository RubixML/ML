<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Backends;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Tasks\Task;
use PHPUnit\Framework\TestCase;

#[Group('Backends')]
#[CoversClass(Serial::class)]
class SerialTest extends TestCase
{
    protected Serial $backend;

    /**
     * @param int $i
     * @return array<int|float>
     */
    public static function foo(int $i) : array
    {
        return [$i * 2, microtime(true)];
    }

    protected function setUp() : void
    {
        $this->backend = new Serial();
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
    }
}
