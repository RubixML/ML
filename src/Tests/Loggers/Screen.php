<?php

namespace Rubix\Engine\Tests\Loggers;

class Screen implements Logger
{
    const DATE_FORMAT = 'Y-m-d H:i:s';

    /**
     * Log the message to the screen.
     *
     * @param  string  $message
     * @return void
     */
    public function log(string $message) : void
    {
        echo (string) date(self::DATE_FORMAT) . ' - ' . trim($message) . "\n";
    }
}
