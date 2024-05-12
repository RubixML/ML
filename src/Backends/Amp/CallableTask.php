<?php

namespace Rubix\ML\Backends\Amp;

use Amp\Cancellation;
use Amp\Parallel\Worker\Task;
use Amp\Sync\Channel;

/**
 * @template-covariant TResult
 * @template TReceive
 * @template TSend
 * @implements Task<TResult, TReceive, TSend>
 */
class CallableTask implements Task
{
    /**
     * @var callable
     */
    private $callable;

    public function __construct(callable $callable)
    {
        $this->callable = $callable;
    }

    public function run(Channel $channel, Cancellation $cancellation) : mixed
    {
        return ($this->callable)();
    }
}
