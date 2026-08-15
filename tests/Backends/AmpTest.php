<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Backends;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Tasks\Task;
use PHPUnit\Framework\TestCase;

#[Group('Backends')]
#[CoversClass(Amp::class)]
class AmpTest extends TestCase
{
    protected Amp $backend;

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
        $this->backend = new Amp(4);
    }

    public function testWorkers() : void
    {
        $this->assertEquals(4, $this->backend->workers());
    }

    public function testEnqueueProcess() : void
    {
        for ($i = 0; $i < 10; ++$i) {
            $this->backend->enqueue(
                task: new Task(fn: [self::class, 'foo'], args: [$i])
            );
        }

        $results = $this->backend->process();

        $this->assertCount(10, $results);
    }
}
