<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers\Schedulers;

use Rubix\ML\NeuralNet\Optimizers\Schedulers\Scheduler;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Cyclical;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Schedulers')]
#[CoversClass(Cyclical::class)]
class CyclicalTest extends TestCase
{
    /**
     * The Cyclical schedule must implement the Scheduler contract so the
     * network can pair it with any Optimizer and advance it once per
     * batch, independent of the number of parameters.
     */
    #[Test]
    public function implementsSchedulerContract() : void
    {
        $scheduler = new Cyclical();

        $this->assertInstanceOf(Scheduler::class, $scheduler);
    }

    /**
     * The schedule's internal counter must advance by one per batch, not
     * one per parameter. A network with K trainable parameters performs
     * K `step()` calls per batch but only one `tick()`.
     */
    #[Test]
    public function scheduleTicksPerBatchNotPerParameter() : void
    {
        $scheduler = new Cyclical(0.001, 0.006, 2, 0.5);

        $K = 3;
        $B = 3;

        $expected = function (int $t) : float {
            $lower = 0.001;
            $upper = 0.006;
            $range = $upper - $lower;
            $length = 2;
            $decay = 0.5;

            $cycle = floor(1 + $t / (2 * $length));
            $x = abs($t / $length - 2 * $cycle + 1);
            $scale = $decay ** $t;

            return $lower + $range * max(0, 1 - $x) * $scale;
        };

        for ($batch = 0; $batch < $B; ++$batch) {
            $expectedRate = $expected($batch);

            for ($i = 0; $i < $K; ++$i) {
                $this->assertEqualsWithDelta($expectedRate, $scheduler->rate(), 1e-12);
            }

            $scheduler->tick();
        }

        $this->assertEqualsWithDelta($expected($B), $scheduler->rate(), 1e-12);
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $scheduler = new Cyclical();

        $this->assertEquals('Cyclical (lower: 0.001, upper: 0.006, length: 2000, decay: 0.99994)', (string) $scheduler);
    }
}
