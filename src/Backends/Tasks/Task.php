<?php

namespace Rubix\ML\Backends\Tasks;

use Amp\Cancellation;
use Rubix\ML\Deferred;
use Amp\Parallel\Worker\Task as AmpTask;
use Amp\Sync\Channel;

/**
 * Task
 *
 * A deferred computation that can be enqueued and processed by a backend.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Task extends Deferred implements AmpTask
{
    /**
     * Run the task in a worker process.
     *
     * @param Channel $channel
     * @param Cancellation $cancellation
     * @return mixed
     */
    public function run(Channel $channel, Cancellation $cancellation) : mixed
    {
        return $this->compute();
    }
}
