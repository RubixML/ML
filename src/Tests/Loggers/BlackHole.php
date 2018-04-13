<?php

namespace Rubix\Engine\Tests\Loggers;

class BlackHole implements Logger
{
    /**
     * Send the message into oblivion.
     *
     * @param  string  $message
     * @return void
     */
    public function log(string $message) : void
    {
        /**
        *    ⬤
        */
    }
}
