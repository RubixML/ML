<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers\Schedulers;

use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Schedulers')]
#[CoversClass(Constant::class)]
class ConstantTest extends TestCase
{
    /**
     * The Constant schedule must implement the Scheduler contract so the
     * network can pair it with any Optimizer and safe to advance it
     * once per batch, even though the rate itself never changes.
     */
    #[Test]
    public function implementsSchedulerContract() : void
    {
        $scheduler = new Constant(0.001);

        $this->assertInstanceOf(Scheduler::class, $scheduler);
    }

    #[Test]
    public function returnsConstantRate() : void
    {
        $scheduler = new Constant(0.001);

        $this->assertEquals(0.001, $scheduler->rate());

        for ($i = 0; $i < 100; ++$i) {
            $scheduler->tick();
        }

        $this->assertEquals(0.001, $scheduler->rate());
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $scheduler = new Constant(0.001);

        $this->assertEquals('Constant (rate: 0.001)', (string) $scheduler);
    }

    #[Test]
    public function defaults() : void
    {
        $scheduler = new Constant();

        $this->assertEquals(0.01, $scheduler->rate());
    }

    #[Test]
    public function badRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('Learning rate must be greater than 0, -1 given.');

        new Constant(-1);
    }
}
