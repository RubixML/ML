<?php

namespace Rubix\ML\Tests\DataProvider;

use Generator;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Swoole;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\SwooleExtensionIsLoaded;

trait BackendProviderTrait
{
    /**
     * @return Generator<string,array<Backend>>
     */
    public static function provideBackends() : Generator
    {
        $serialBackend = new Serial();

        yield (string) $serialBackend => [
            'backend' => $serialBackend,
        ];

        // $ampBackend = new Amp();

        // yield (string) $ampBackend => [
        //     'backend' => $ampBackend,
        // ];

        if (
            SwooleExtensionIsLoaded::create()->passes()
            && ExtensionIsLoaded::with('igbinary')->passes()
        ) {
            $swooleProcessBackend = new Swoole();

            yield (string) $swooleProcessBackend => [
                'backend' => $swooleProcessBackend,
            ];
        }
    }
}
