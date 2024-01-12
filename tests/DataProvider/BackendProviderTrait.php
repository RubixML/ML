<?php

namespace Rubix\ML\Tests\DataProvider;

use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Swoole;
use Rubix\ML\Specifications\SwooleExtensionIsLoaded;

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

        $ampBackend = new Amp();
        $backends[(string) $ampBackend] = [$ampBackend];

        if (SwooleExtensionIsLoaded::create()->passes()) {
            $swooleProcessBackend = new Swoole();
            $backends[(string) $swooleProcessBackend] = [$swooleProcessBackend];
        }

        return $backends;
    }
}
