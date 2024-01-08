<?php

namespace Rubix\ML\Tests\DataProvider;

use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Amp as AmpBackend;
use Rubix\ML\Backends\Swoole\Coroutine as SwooleCoroutineBackend;
use Rubix\ML\Backends\Swoole\Process as SwooleProcessBackend;

trait BackendProviderTrait
{
    /**
     * @return array<Backend>
     */
    public static function provideBackends() : array
    {
        $backends = [];

        $serialBackend = new Serial();
        $backends[(string) $serialBackend] = [$serialBackend];

        $ampBackend = new AmpBackend();
        $backends[(string) $ampBackend] = [$ampBackend];

        if (extension_loaded('swoole') || extension_loaded('openswoole')) {
            $swooleCoroutineBackend = new SwooleCoroutineBackend();
            $backends[(string) $swooleCoroutineBackend] = [$swooleCoroutineBackend];

            $swooleProcessBackend = new SwooleProcessBackend();
            $backends[(string) $swooleProcessBackend] = [$swooleProcessBackend];
        }

        return $backends;
    }
}
