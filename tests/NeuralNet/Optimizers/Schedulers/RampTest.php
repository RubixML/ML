<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers\Schedulers;

use Rubix\ML\NeuralNet\Optimizers\Schedulers\Ramp;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Schedulers')]
#[CoversClass(Ramp::class)]
class RampTest extends TestCase
{
    /**
     * The Ramp schedule must implement the Scheduler contract so the network
     * can pair it with any Optimizer and advance it once per batch,
     * independent of the number of parameters.
     */
    #[Test]
    public function implementsSchedulerContract() : void
    {
        $scheduler = new Ramp();

        $this->assertInstanceOf(Scheduler::class, $scheduler);
    }

    /**
     * The rate must ramp linearly from the start rate to the end rate,
     * reaching exactly the end rate once the step budget is exhausted.
     */
    #[Test]
    public function rateRampsLinearly() : void
    {
        $start = 0.001;
        $end = 0.01;
        $steps = 100;
        $delta = $end - $start;

        $scheduler = new Ramp($start, $end, $steps);

        $this->assertEqualsWithDelta($start, $scheduler->rate(), 1e-12);

        for ($t = 1; $t <= $steps; ++$t) {
            $scheduler->tick();

            $expected = $start + $delta * ($t / $steps);

            $this->assertEqualsWithDelta($expected, $scheduler->rate(), 1e-12);
        }

        $this->assertEqualsWithDelta($end, $scheduler->rate(), 1e-12);
    }

    /**
     * Once the step budget is exhausted the rate must hold at the end rate
     * for the remainder of training, regardless of the direction of the ramp.
     */
    #[Test]
    public function rateHoldsAtEndRate() : void
    {
        $start = 0.001;
        $end = 0.01;
        $steps = 10;

        $scheduler = new Ramp($start, $end, $steps);

        for ($t = 0; $t < $steps; ++$t) {
            $scheduler->tick();
        }

        $this->assertEqualsWithDelta($end, $scheduler->rate(), 1e-12);

        for ($t = 0; $t < 100; ++$t) {
            $scheduler->tick();
        }

        $this->assertEqualsWithDelta($end, $scheduler->rate(), 1e-12);
    }

    /**
     * The schedule's internal counter must advance by one per batch, not
     * one per parameter. A network with K trainable parameters performs
     * K `step()` calls per batch but only one `tick()`.
     */
    #[Test]
    public function scheduleTicksPerBatchNotPerParameter() : void
    {
        $start = 0.001;
        $end = 0.01;
        $steps = 100;
        $delta = $end - $start;

        $scheduler = new Ramp($start, $end, $steps);

        $K = 3;
        $B = 3;

        for ($batch = 0; $batch < $B; ++$batch) {
            $expected = $start + $delta * ($batch / $steps);

            for ($i = 0; $i < $K; ++$i) {
                $this->assertEqualsWithDelta($expected, $scheduler->rate(), 1e-12);
            }

            $scheduler->tick();
        }

        $this->assertEqualsWithDelta($start + $delta * ($B / $steps), $scheduler->rate(), 1e-12);
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $scheduler = new Ramp(0.001, 0.01, 1000);

        $this->assertEquals('Ramp (start: 0.001, end: 0.01, steps: 1000)', (string) $scheduler);
    }

    #[Test]
    public function defaults() : void
    {
        $scheduler = new Ramp();

        $this->assertEqualsWithDelta(0.001, $scheduler->rate(), 1e-12);
    }

    #[Test]
    public function badStartRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('Starting learning rate must be greater than 0, -1 given.');

        new Ramp(-1);
    }

    #[Test]
    public function badEndRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('Ending learning rate must be greater than 0, -1 given.');

        new Ramp(0.001, -1);
    }

    #[Test]
    public function badSteps() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('The number of steps must be greater than 0, 0 given.');

        new Ramp(0.001, 0.01, 0);
    }
}
