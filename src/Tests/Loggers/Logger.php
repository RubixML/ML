<?php

namespace Rubix\Engine\Tests\Loggers;

interface Logger
{
    /**
     * Log the message.
     *
     * @param  string  $message
     * @return void
     */
    public function log(string $message) : void;
}
