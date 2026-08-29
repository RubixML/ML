<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Backends;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Tasks\Task;
use PHPUnit\Framework\TestCase;

#[Group('Backends')]
#[CoversClass(Amp::class)]
class AmpTest extends TestCase
{
    protected Amp $backend;

    protected ?Backend $usedBackend = null;

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

    protected function tearDown() : void
    {
        $this->usedBackend?->shutdown();
    }

    #[Test]
    public function workers() : void
    {
        $this->assertEquals(4, $this->backend->workers());
    }

    #[Test]
    public function enqueueProcess() : void
    {
        $this->usedBackend = $this->backend;

        for ($i = 0; $i < 10; ++$i) {
            $this->backend->enqueue(
                task: new Task(fn: [self::class, 'foo'], args: [$i])
            );
        }

        $results = $this->backend->process();

        $this->assertCount(10, $results);
    }
}
