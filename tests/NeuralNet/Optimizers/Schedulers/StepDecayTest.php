<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers\Schedulers;

use Rubix\ML\NeuralNet\Optimizers\Schedulers\StepDecay;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Schedulers')]
#[CoversClass(StepDecay::class)]
class StepDecayTest extends TestCase
{
    /**
     * The Step Decay schedule must implement the Scheduler contract so the
     * network can pair it with any Optimizer and advance it once per batch,
     * independent of the number of parameters.
     */
    #[Test]
    public function implementsSchedulerContract() : void
    {
        $scheduler = new StepDecay();

        $this->assertInstanceOf(Scheduler::class, $scheduler);
    }

    /**
     * The schedule's internal counter must advance by exactly one per tick,
     * not once per parameter. This guarantees that a network with K trainable
     * parameters performs K `step()` calls per batch but the schedule only
     * advances once per batch so the rate is consistent across all parameters.
     */
    #[Test]
    public function rateDecaysPerFloor() : void
    {
        $rate = 0.01;
        $losses = 1;
        $decay = 0.5;

        $scheduler = new StepDecay($rate, $losses, $decay);

        $expected = [
            0.01,
            0.01 / (1.0 + 1 * $decay),
            0.01 / (1.0 + 2 * $decay),
        ];

        foreach ($expected as $i => $expectedRate) {
            $this->assertEqualsWithDelta($expectedRate, $scheduler->rate(), 1e-12);

            $scheduler->tick();
        }
    }

    /**
     * The schedule's internal counter must advance by one per batch, not
     * one per parameter. After B batches the rate must reflect B ticks.
     */
    #[Test]
    public function rateReflectsNumberOfBatches() : void
    {
        $rate = 0.01;
        $losses = 1;
        $decay = 0.5;

        $scheduler = new StepDecay($rate, $losses, $decay);

        $K = 3;
        $B = 3;

        $expectedBatch = [
            0.01,
            0.01 / (1.0 + 1 * $decay),
            0.01 / (1.0 + 2 * $decay),
        ];

        foreach ($expectedBatch as $expected) {
            for ($i = 0; $i < $K; ++$i) {
                $this->assertEqualsWithDelta($expected, $scheduler->rate(), 1e-12);
            }

            $scheduler->tick();
        }

        $this->assertEqualsWithDelta(0.01 / (1.0 + $B * $decay), $scheduler->rate(), 1e-12);
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $scheduler = new StepDecay(0.01, 100, 1e-3);

        $this->assertEquals('Step Decay (rate: 0.01, steps: 100, decay: 0.001)', (string) $scheduler);
    }
}
