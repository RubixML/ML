<?php

namespace Rubix\ML\Tests\NeuralNet\Optimizers;

use Rubix\ML\NeuralNet\Optimizers\Cyclical;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Scheduler;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\StepDecay;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Optimizers')]
#[CoversClass(Scheduler::class)]
class SchedulerTest extends TestCase
{
    /**
     * The global learning-rate schedulers must implement the
     * Scheduler contract so the network can advance them once
     * per batch, independent of the number of parameters.
     */
    #[Test]
    public function stepDecayIsAScheduler() : void
    {
        $optimizer = new StepDecay();

        $this->assertInstanceOf(Optimizer::class, $optimizer);
        $this->assertInstanceOf(Scheduler::class, $optimizer);
    }

    #[Test]
    public function cyclicalIsAScheduler() : void
    {
        $optimizer = new Cyclical();

        $this->assertInstanceOf(Optimizer::class, $optimizer);
        $this->assertInstanceOf(Scheduler::class, $optimizer);
    }

    /**
     * Stateless optimizers such as Stochastic are not schedulers
     * (they have no schedule to advance) so the FeedForward
     * network can safely skip them.
     */
    #[Test]
    public function stochasticIsNotAScheduler() : void
    {
        $optimizer = new Stochastic();

        $this->assertInstanceOf(Optimizer::class, $optimizer);
        $this->assertNotInstanceOf(Scheduler::class, $optimizer);
    }
}
