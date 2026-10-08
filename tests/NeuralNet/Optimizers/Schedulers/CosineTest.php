<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers\Schedulers;

use Rubix\ML\NeuralNet\Optimizers\Schedulers\Cosine;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Schedulers')]
#[CoversClass(Cosine::class)]
class CosineTest extends TestCase
{
    /**
     * The Cosine schedule must implement the Scheduler contract so the
     * network can pair it with any Optimizer and advance it once per batch,
     * independent of the number of parameters.
     */
    #[Test]
    public function implementsSchedulerContract() : void
    {
        $scheduler = new Cosine();

        $this->assertInstanceOf(Scheduler::class, $scheduler);
    }

    /**
     * The rate must decay smoothly and monotonically from the start rate
     * down to the end rate, reaching exactly the end rate once the step
     * budget is exhausted.
     */
    #[Test]
    public function rateDecaysCosinely() : void
    {
        $start = 0.01;
        $end = 0.0001;
        $steps = 100;

        $scheduler = new Cosine($start, $end, $steps);

        $this->assertEqualsWithDelta($start, $scheduler->rate(), 1e-12);

        $previous = $start;

        for ($t = 1; $t <= $steps; ++$t) {
            $scheduler->tick();

            $expected = $end + ($start - $end) * (1 + cos(M_PI * $t / $steps)) / 2;

            $this->assertEqualsWithDelta($expected, $scheduler->rate(), 1e-12);

            $this->assertLessThanOrEqual($previous, $scheduler->rate());

            $previous = $scheduler->rate();
        }

        $this->assertEqualsWithDelta($end, $scheduler->rate(), 1e-12);
    }

    /**
     * Once the step budget is exhausted the rate must hold at the end rate
     * for the remainder of training.
     */
    #[Test]
    public function rateHoldsAtEndRate() : void
    {
        $start = 0.01;
        $end = 0.0001;
        $steps = 10;

        $scheduler = new Cosine($start, $end, $steps);

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
        $start = 0.01;
        $end = 0.0001;
        $steps = 100;

        $scheduler = new Cosine($start, $end, $steps);

        $K = 3;
        $B = 3;

        for ($batch = 0; $batch < $B; ++$batch) {
            $expected = $end + ($start - $end) * (1 + cos(M_PI * $batch / $steps)) / 2;

            for ($i = 0; $i < $K; ++$i) {
                $this->assertEqualsWithDelta($expected, $scheduler->rate(), 1e-12);
            }

            $scheduler->tick();
        }

        $this->assertEqualsWithDelta(
            $end + ($start - $end) * (1 + cos(M_PI * $B / $steps)) / 2,
            $scheduler->rate(),
            1e-12
        );
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $scheduler = new Cosine(0.01, 0.0001, 1000);

        $this->assertEquals('Cosine (start: 0.01, end: 0.0001, steps: 1000)', (string) $scheduler);
    }

    #[Test]
    public function defaults() : void
    {
        $scheduler = new Cosine();

        $this->assertEqualsWithDelta(0.01, $scheduler->rate(), 1e-12);
    }

    #[Test]
    public function badStart() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('Starting rate must be greater than 0, -1 given.');

        new Cosine(-1);
    }

    #[Test]
    public function badEnd() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('The ending rate must be greater than 0, -1 given.');

        new Cosine(0.01, -1);
    }

    #[Test]
    public function badSteps() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->expectExceptionMessage('The number of steps must be greater than 0, 0 given.');

        new Cosine(0.01, 0.0001, 0);
    }
}
