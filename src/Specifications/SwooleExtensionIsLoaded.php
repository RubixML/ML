<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Exceptions\MissingExtension;

/**
 * @internal
 */
class SwooleExtensionIsLoaded extends Specification
{
    public static function create() : self
    {
        return new self();
    }

    /**
     * @throws \Rubix\ML\Exceptions\MissingExtension
     */
    public function check() : void
    {
        if (
            ExtensionIsLoaded::with('swoole')->passes()
            || ExtensionIsLoaded::with('openswoole')->passes()
        ) {
            return;
        }

        throw new MissingExtension('swoole');
    }
}
